"""Schedules and time propagation for the chain model.

dpsi/dt = -2 pi i (H/h) psi with H/h in GHz and t in ns. Columns are integrated
in the frame rotating at their own dressed idle energy, exactly as in the
two-qubit kernel, and rotated back afterwards.

Flux-only schedules conserve total local-index parity exactly (n_i n_j and cos
terms), so both parity blocks are integrated as one concatenated DOP853 state,
as in the inherited two-qubit algorithm. Any microwave term breaks parity and
selects full-space DOP853 with a carrier-resolving maximum step and restarts at
envelope edges. Each coupler's time-dependent scalar -(dEJ) cos[0,0] is removed
as a global phase, as in the inherited algorithm.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
from scipy.integrate import solve_ivp

from .model import squid_EJ_GHz, _real, digest


# Exact copies of emulateqpu_twoq.core._kernel.commanded_flux / delivered_flux.
def commanded_flux(t, amplitude, command_duration, *, idle=0., rise_fraction=.2):
    """Raised-cosine rise/hold/fall; rise_fraction=.5 is a sin² pulse."""
    u = np.clip(np.asarray(t, float), 0, command_duration); rise = command_duration*rise_fraction
    value = np.where(u < rise, .5*(1-np.cos(np.pi*u/rise)),
        np.where(u <= command_duration-rise, 1., .5*(1+np.cos(np.pi*(u-command_duration+rise)/rise))))
    return idle+amplitude*value


def delivered_flux(t, amplitude, duration, *, idle=0., tau_ns=1., gain=.98, settling=5., rise_fraction=.5):
    """Causal first-order low-pass response to a finite-slope excursion (assumed line)."""
    t = np.asarray(t, float); command_duration = duration-tau_ns*settling
    if command_duration <= 0: raise ValueError('No positive commanded pulse interval remains')
    if tau_ns == 0: return commanded_flux(t, gain*amplitude, command_duration, idle=idle, rise_fraction=rise_fraction)
    rise = command_duration*rise_fraction; hold = command_duration-2*rise; w = np.pi/rise
    def segment(u, x0, c0, c1, frequency):
        decay = np.exp(-u/tau_ns); a = frequency*tau_ns
        return x0*decay+gain*(c0*(1-decay)+c1*(np.cos(frequency*u)+a*np.sin(frequency*u)-decay)/(1+a*a))
    xrise = segment(rise, 0., amplitude/2, -amplitude/2, w)
    xhold = segment(hold, xrise, amplitude, 0., 0.)
    xfall = segment(rise, xhold, amplitude/2, amplitude/2, w)
    answer = np.where(t < rise, segment(np.maximum(t, 0), 0., amplitude/2, -amplitude/2, w),
        np.where(t < rise+hold, segment(np.clip(t-rise, 0, hold), xrise, amplitude, 0., 0.),
            np.where(t <= command_duration, segment(np.clip(t-rise-hold, 0, rise), xhold, amplitude/2, amplitude/2, w),
                xfall*np.exp(-np.maximum(t-command_duration, 0)/tau_ns))))
    return idle+answer


ENVELOPES = ('square', 'cosine', 'flattop')


def validate_schedule(model, schedule):
    if not isinstance(schedule, dict): raise ValueError('Schedule object required')
    s = deepcopy(schedule)
    allowed = {'duration_ns', 'flux', 'drives', 'output_phases_rad', 'target', 'purpose'}
    if set(s)-allowed: raise ValueError('Unknown schedule fields: '+str(sorted(set(s)-allowed)))
    labels = model.cell['mode_labels']
    s['duration_ns'] = _real(s['duration_ns'], 'duration', .01, 2000.)
    T = s['duration_ns']
    flux = s.setdefault('flux', [])
    if not isinstance(flux, list): raise ValueError('flux must be a list of coupler pulses')
    seen = set()
    for f in flux:
        if not isinstance(f, dict) or set(f)-{'coupler', 'amplitude_Phi0', 'start_ns', 'duration_ns'}: raise ValueError('Invalid flux pulse')
        if f.get('coupler') not in labels or labels.index(f['coupler']) not in model.coupler_modes: raise ValueError('Flux pulse needs a coupler label')
        if f['coupler'] in seen: raise ValueError('At most one flux pulse per coupler')
        seen.add(f['coupler'])
        i = labels.index(f['coupler']); line = model.cell['lines'][f['coupler']]; idle = model.cell['idle_flux_Phi0'][i]
        f['amplitude_Phi0'] = _real(f['amplitude_Phi0'], 'flux amplitude', -.47, .47)
        f['start_ns'] = _real(f.get('start_ns', 0.), 'flux start', 0., T)
        f['duration_ns'] = _real(f.get('duration_ns', T-f['start_ns']), 'flux duration', .01, 2000.)
        if f['start_ns']+f['duration_ns'] > T+1e-12 or f['duration_ns'] <= line['tau_ns']*line['settling_time_constants']:
            raise ValueError('Flux pulse with its settling time must fit inside the schedule')
        if not 0 <= idle+line['gain']*f['amplitude_Phi0'] <= .47: raise ValueError('Delivered flux may leave the supported SQUID branch')
    drives = s.setdefault('drives', [])
    if not isinstance(drives, list) or len(drives) > 64: raise ValueError('At most 64 microwave pulses')
    for d in drives:
        if not isinstance(d, dict) or set(d)-{'mode', 'start_ns', 'duration_ns', 'amplitude_GHz', 'frequency_GHz', 'phase_rad', 'envelope', 'ramp_ns'}:
            raise ValueError('Unknown microwave fields')
        if d.get('mode') not in labels: raise ValueError('Drive mode must be a mode label')
        d['start_ns'] = _real(d.get('start_ns', 0.), 'drive start', 0., T)
        d['duration_ns'] = _real(d.get('duration_ns', T-d['start_ns']), 'drive duration', .01, 2000.)
        if d['start_ns']+d['duration_ns'] > T+1e-12: raise ValueError('Microwave exceeds schedule')
        d['amplitude_GHz'] = _real(d['amplitude_GHz'], 'charge coefficient', -.5, .5)
        d['frequency_GHz'] = _real(d['frequency_GHz'], 'carrier', 0., 20.)
        d['phase_rad'] = _real(d.get('phase_rad', 0.), 'carrier phase')
        d.setdefault('envelope', 'cosine')
        if d['envelope'] not in ENVELOPES: raise ValueError('Envelope must be square, cosine (sin^2) or flattop')
        if d['envelope'] == 'flattop':
            d['ramp_ns'] = _real(d.get('ramp_ns', 20.), 'ramp', .01, 1000.)
            if 2*d['ramp_ns'] > d['duration_ns']: raise ValueError('Flat-top ramps exceed pulse duration')
        elif 'ramp_ns' in d: raise ValueError('ramp_ns applies only to flattop')
    nd = len(model.data_modes)
    phases = s.setdefault('output_phases_rad', [0.]*nd)
    if not isinstance(phases, list) or len(phases) != nd: raise ValueError('One output phase per data qubit')
    s['output_phases_rad'] = [_real(p, 'output phase') for p in phases]
    return s


def _envelope(d, t):
    u = t-d['start_ns']
    if u < 0 or u > d['duration_ns']: return 0.
    if d['envelope'] == 'square': return 1.
    if d['envelope'] == 'cosine': return float(np.sin(np.pi*u/d['duration_ns'])**2)
    r = d['ramp_ns']
    if u < r: return float(np.sin(.5*np.pi*u/r)**2)
    if u > d['duration_ns']-r: return float(np.sin(.5*np.pi*(d['duration_ns']-u)/r)**2)
    return 1.


def controls_at_time(model, s, t):
    """Return {coupler mode: delivered flux}, {mode: microwave coefficient GHz}."""
    labels = model.cell['mode_labels']
    phi = {i: model.cell['idle_flux_Phi0'][i] for i in model.coupler_modes}
    for f in s['flux']:
        i = labels.index(f['coupler']); line = model.cell['lines'][f['coupler']]
        if t >= f['start_ns']:
            phi[i] = float(delivered_flux(t-f['start_ns'], f['amplitude_Phi0'], f['duration_ns'], idle=model.cell['idle_flux_Phi0'][i],
                tau_ns=line['tau_ns'], gain=line['gain'], settling=line['settling_time_constants'], rise_fraction=line['rise_fraction']))
    mw = {}
    for d in s['drives']:
        e = _envelope(d, t)
        if e:
            i = labels.index(d['mode'])
            mw[i] = mw.get(i, 0.)+d['amplitude_GHz']*e*np.cos(2*np.pi*d['frequency_GHz']*t+d['phase_rad'])
    return phi, mw


def _real_spmm(A, X):
    """Real CSR times complex dense without complex sparse copies."""
    X = np.ascontiguousarray(X); D, k = X.shape
    Xr = X.view(np.float64).reshape(D, 2*k)
    return np.ascontiguousarray(A@Xr).view(np.complex128).reshape(D, k)


def propagate(model, schedule, *, inputs=None, rtol=1e-9, atol=1e-11, samples=3, keep_states=False):
    """Propagate dressed computational inputs (default all 2^n) through a schedule."""
    s = validate_schedule(model, schedule)
    T = s['duration_ns']; times = np.linspace(0, T, samples)
    enc = model.encoding; E = model.encoded_energies; ncomp = enc.shape[1]
    inputs = list(range(ncomp)) if inputs is None else list(inputs)
    flux_couplers = [model.cell['mode_labels'].index(f['coupler']) for f in s['flux']]
    C = {i: model.cos_ops[i]-model.local[i][2][0, 0]*_eye(model) for i in flux_couplers}
    EJ = model.EJ; EJmax = {i: model.EJmax(i) for i in model.coupler_modes}

    def deltas(t):
        phi, mw = controls_at_time(model, s, t)
        return {i: float(squid_EJ_GHz(EJmax[i], phi[i])-EJ[i]) for i in flux_couplers}, mw

    started = time.perf_counter()
    if not s['drives']:
        bits = np.array(list(np.ndindex((2,)*len(model.data_modes))))
        cpar = bits.sum(axis=1) % 2
        blocks = []; cursor = 0; forbidden = 0.
        for p in (0, 1):
            cols = [k for k in inputs if cpar[k] == p]
            if not cols: continue
            rows = np.flatnonzero(model.basis.parity == p)
            forbidden = max(forbidden, float(np.abs(enc[np.ix_(model.basis.parity != p, cols)]).max()))
            B = model.H[rows][:, rows].tocsr()
            Cb = {i: C[i][rows][:, rows].tocsr() for i in flux_couplers}
            V = enc[np.ix_(rows, cols)]
            blocks.append((rows, cols, B, Cb, V, slice(cursor, cursor+V.size)))
            cursor += V.size
        if forbidden > 1e-9: raise ValueError('Idle encoding parity assignment is ambiguous')

        def rhs(t, y):
            dl, _ = deltas(t)
            out = np.empty_like(y)
            for rows, cols, B, Cb, V, sl in blocks:
                X = y[sl].reshape(V.shape)
                Z = _real_spmm(B, X)
                for i, dv in dl.items():
                    if dv: Z -= dv*_real_spmm(Cb[i], X)
                Z -= X*E[cols]
                out[sl] = (-2j*np.pi*Z).ravel()
            return out
        y0 = np.concatenate([b[4].ravel() for b in blocks])
        sol = solve_ivp(rhs, (0, T), y0, t_eval=times, method='DOP853', rtol=rtol, atol=atol)
        if not sol.success: raise RuntimeError(sol.message)
        Y = np.zeros((samples, model.basis.dimension, len(inputs)), complex)
        pos = {k: j for j, k in enumerate(inputs)}
        for rows, cols, B, Cb, V, sl in blocks:
            vals = sol.y[sl].T.reshape(samples, *V.shape)
            vals = vals*np.exp(-2j*np.pi*times[:, None, None]*E[cols])
            for j, k in enumerate(cols): Y[:, rows, pos[k]] = vals[:, :, j]
        nfev = int(sol.nfev); engine = 'parity-block flux-only DOP853 (inherited algorithm, sparse operators)'
    else:
        Xshape = (model.basis.dimension, len(inputs))
        H = model.H; N = model.charge_ops; Ein = E[inputs]

        def rhs(t, y):
            dl, mw = deltas(t)
            X = y.reshape(Xshape)
            Z = _real_spmm(H, X)
            for i, dv in dl.items():
                if dv: Z -= dv*_real_spmm(C[i], X)
            for i, a in mw.items():
                if a: Z += a*_real_spmm(N[i], X)
            Z -= X*Ein
            return (-2j*np.pi*Z).ravel()
        carrier = max(d['frequency_GHz'] for d in s['drives'])
        max_step = min(T, 1/(16*carrier)) if carrier > 0 else T
        edges = sorted(set([0., T]+[v for d in s['drives'] for v in (d['start_ns'], d['start_ns']+d['duration_ns'])]+
                           [v for d in s['drives'] if d['envelope'] == 'flattop' for v in (d['start_ns']+d['ramp_ns'], d['start_ns']+d['duration_ns']-d['ramp_ns'])]))
        Y = np.empty((samples, *Xshape), complex); Y[0] = enc[:, inputs]; y = Y[0].ravel().copy(); nfev = 0
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = np.flatnonzero((times > lo) & (times <= hi))
            req = times[idx]
            ev = req if len(req) and req[-1] == hi else np.r_[req, hi]
            sol = solve_ivp(rhs, (lo, hi), y, method='DOP853', rtol=rtol, atol=atol, max_step=max_step, t_eval=ev)
            if not sol.success: raise RuntimeError(sol.message)
            Y[idx] = sol.y[:, :len(idx)].T.reshape(len(idx), *Xshape)
            y = sol.y[:, -1]; nfev += int(sol.nfev)
        Y *= np.exp(-2j*np.pi*times[:, None, None]*Ein)
        engine = 'full-space parity-breaking lab-frame DOP853; no RWA'
    elapsed = time.perf_counter()-started
    encoded = np.einsum('dk,tdj->tkj', enc.conj(), Y)
    pop = np.abs(Y)**2
    S = model.basis.states
    data_mask = np.any(S[:, list(model.data_modes)] >= 2, axis=1).astype(float)
    coupler_mask = (np.any(S[:, list(model.coupler_modes)] > 0, axis=1) if model.coupler_modes else np.zeros(len(S), bool)).astype(float)
    top = (model.basis.weighted_excitations == (model.basis.cap if model.basis.cap is not None else -1)).astype(float)
    norm_err = max(float(np.abs(x.conj().T@x-np.eye(len(inputs))).max()) for x in Y)
    out = dict(times_ns=times, encoded=encoded, final=Y[-1], inputs=inputs,
        bare_data_leakage=np.einsum('tdk,d->tk', pop, data_mask), bare_coupler_excitation=np.einsum('tdk,d->tk', pop, coupler_mask),
        cap_boundary_population=np.einsum('tdk,d->tk', pop, top), norm_error=norm_err, nfev=nfev, elapsed_s=elapsed, engine=engine,
        schedule=s, schedule_sha256=digest(s), rtol=rtol, atol=atol)
    if keep_states: out['states'] = Y
    return out


def _eye(model):
    import scipy.sparse as sp
    return sp.identity(model.basis.dimension, format='csr')
