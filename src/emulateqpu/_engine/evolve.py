"""Pulse library and propagation of arbitrary state vectors (0.1.0.dev4).

New module; propagate.py is unchanged and remains the reference for dressed
computational inputs. This module adds:

- schedule v2 (`emulateqpu.chain.schedule.v2`):
  - several flux pulses per coupler; the line model is linear, so the delivered flux is
    idle + the sum of each pulse's line response. Shapes:
    - `raised_cosine`: the existing command and line response (propagate.delivered_flux);
    - `sampled`: piecewise-linear command samples through the same first-order line, solved
      exactly per segment.
  - microwave envelopes: square, cosine, flattop (as before), plus gaussian, drag and sampled.
- propagate_states(model, schedule, V): evolves any D x r block of state vectors.

Drive convention, unchanged for the old envelopes:
  coefficient(t) = a [I(t) cos(2 pi f (t + t0) + phi) + Q(t) sin(2 pi f (t + t0) + phi)],
which multiplies the mode's charge operator n. Here t0 = carrier_time_offset_ns, the absolute program clock at
the segment start; with t0 = 0 the result is identical to propagate.py. In the RWA picture this is
H_rf = (a n01 / 2) [I X + Q Y] at phi = 0. So Q is the in-frame Y quadrature.

DRAG: I is the lifted Gaussian, and Q = drag_coefficient_ns * dI/dt. The first-order transmon value is about
-1/(2 pi alpha_GHz); it is a user input and is never chosen automatically.

Frames: each column is integrated in a frame rotating at its own Rayleigh-quotient energy, then rotated back. This
is only a numerical frame; the returned states are Schrodinger-picture vectors in the retained product basis, as in
propagate.py.

Numerical model only; no hardware pulse distortion beyond the declared first-order line.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
from scipy.integrate import solve_ivp

from .model import squid_EJ_GHz, _real, digest
from .propagate import delivered_flux, _real_spmm, _eye

SCHEDULE_SCHEMA = 'emulateqpu.chain.schedule.v2'
ENVELOPES = ('square', 'cosine', 'flattop', 'gaussian', 'drag', 'sampled')
FLUX_SHAPES = ('raised_cosine', 'sampled')
MAX_SAMPLES = 20000
CARRIER_STEPS_PER_CYCLE = 16


def _samples(x, name, lo=-1., hi=1.):
    a = np.asarray(x, float)
    if a.ndim != 1 or not 2 <= len(a) <= MAX_SAMPLES or not np.isfinite(a).all() or np.any(a < lo) or np.any(a > hi):
        raise ValueError(name+f' must be 2..{MAX_SAMPLES} finite samples in [{lo}, {hi}]')
    return a.tolist()


def validate_schedule_v2(model, schedule):
    if not isinstance(schedule, dict): raise ValueError('Schedule object required')
    s = deepcopy(schedule)
    allowed = {'schema', 'duration_ns', 'flux', 'drives', 'output_phases_rad', 'target', 'purpose'}
    if set(s)-allowed: raise ValueError('Unknown schedule fields: '+str(sorted(set(s)-allowed)))
    s['schema'] = SCHEDULE_SCHEMA
    labels = model.cell['mode_labels']
    s['duration_ns'] = _real(s['duration_ns'], 'duration', 0., 20000.)
    T = s['duration_ns']
    flux = s.setdefault('flux', [])
    if not isinstance(flux, list) or len(flux) > 64: raise ValueError('flux must be a list of at most 64 coupler pulses')
    for f in flux:
        if not isinstance(f, dict): raise ValueError('Invalid flux pulse')
        f.setdefault('shape', 'raised_cosine')
        if f['shape'] not in FLUX_SHAPES: raise ValueError('Flux shape must be raised_cosine or sampled')
        keys = {'coupler', 'shape', 'start_ns'} | ({'amplitude_Phi0', 'duration_ns'} if f['shape'] == 'raised_cosine' else {'dt_ns', 'samples_Phi0'})
        if set(f)-keys: raise ValueError('Unknown flux-pulse fields: '+str(sorted(set(f)-keys)))
        if f.get('coupler') not in labels or labels.index(f['coupler']) not in model.coupler_modes: raise ValueError('Flux pulse needs a coupler label')
        line = model.cell['lines'][f['coupler']]
        f['start_ns'] = _real(f.get('start_ns', 0.), 'flux start', 0., T)
        if f['shape'] == 'raised_cosine':
            f['amplitude_Phi0'] = _real(f['amplitude_Phi0'], 'flux amplitude', -.47, .47)
            f['duration_ns'] = _real(f.get('duration_ns', T-f['start_ns']), 'flux duration', .01, 20000.)
            if f['duration_ns'] <= line['tau_ns']*line['settling_time_constants']: raise ValueError('Flux pulse shorter than its settling time')
            window = f['duration_ns']
        else:
            f['dt_ns'] = _real(f['dt_ns'], 'flux sample spacing', .01, 100.)
            f['samples_Phi0'] = _samples(f['samples_Phi0'], 'flux samples', -.47, .47)
            if abs(f['samples_Phi0'][0]) > 1e-12 or abs(f['samples_Phi0'][-1]) > 1e-12:
                raise ValueError('Sampled flux commands must start and end at zero excursion')
            window = (len(f['samples_Phi0'])-1)*f['dt_ns']+line['tau_ns']*line['settling_time_constants']
        if f['start_ns']+window > T+1e-9: raise ValueError('Flux pulse with its settling time must fit inside the schedule')
    # The delivered flux of every coupler must stay on the supported SQUID branch.
    grid = np.linspace(0., T, 20001) if T > 0 else np.zeros(1)
    prep = _prepare_flux(model, s)
    for i in model.coupler_modes:
        phi = _delivered_array(model, prep, i, grid)
        if phi.min() < 0. or phi.max() > .47: raise ValueError('Delivered flux leaves the supported SQUID branch on '+labels[i])
    drives = s.setdefault('drives', [])
    if not isinstance(drives, list) or len(drives) > 256: raise ValueError('At most 256 microwave pulses')
    for d in drives:
        if not isinstance(d, dict): raise ValueError('Invalid microwave pulse')
        d.setdefault('envelope', 'cosine')
        if d['envelope'] not in ENVELOPES: raise ValueError('Envelope must be one of '+', '.join(ENVELOPES))
        extra = {'flattop': {'ramp_ns'}, 'gaussian': {'sigma_ns'}, 'drag': {'sigma_ns', 'drag_coefficient_ns'},
                 'sampled': {'samples_I', 'samples_Q'}}.get(d['envelope'], set())
        keys = {'mode', 'start_ns', 'duration_ns', 'amplitude_GHz', 'frequency_GHz', 'phase_rad', 'envelope'} | extra
        if set(d)-keys: raise ValueError('Unknown microwave fields: '+str(sorted(set(d)-keys)))
        if d.get('mode') not in labels: raise ValueError('Drive mode must be a mode label')
        d['start_ns'] = _real(d.get('start_ns', 0.), 'drive start', 0., T)
        d['duration_ns'] = _real(d.get('duration_ns', T-d['start_ns']), 'drive duration', .01, 20000.)
        if d['start_ns']+d['duration_ns'] > T+1e-9: raise ValueError('Microwave exceeds schedule')
        d['amplitude_GHz'] = _real(d['amplitude_GHz'], 'charge coefficient', -.5, .5)
        d['frequency_GHz'] = _real(d['frequency_GHz'], 'carrier', 0., 20.)
        d['phase_rad'] = _real(d.get('phase_rad', 0.), 'carrier phase')
        if d['envelope'] == 'flattop':
            d['ramp_ns'] = _real(d.get('ramp_ns', 20.), 'ramp', .01, 1000.)
            if 2*d['ramp_ns'] > d['duration_ns']: raise ValueError('Flat-top ramps exceed pulse duration')
        if d['envelope'] in ('gaussian', 'drag'):
            d['sigma_ns'] = _real(d.get('sigma_ns', d['duration_ns']/4), 'gaussian sigma', .01, 10000.)
        if d['envelope'] == 'drag':
            d['drag_coefficient_ns'] = _real(d['drag_coefficient_ns'], 'DRAG coefficient', -100., 100.)
        if d['envelope'] == 'sampled':
            d['samples_I'] = _samples(d['samples_I'], 'I samples')
            d['samples_Q'] = _samples(d.get('samples_Q', [0.]*len(d['samples_I'])), 'Q samples')
            if len(d['samples_Q']) != len(d['samples_I']): raise ValueError('I and Q need the same number of samples')
    nd = len(model.data_modes)
    phases = s.setdefault('output_phases_rad', [0.]*nd)
    if not isinstance(phases, list) or len(phases) != nd: raise ValueError('One output phase per data qubit')
    s['output_phases_rad'] = [_real(p, 'output phase') for p in phases]
    return s


def _line_nodes(u, dt, tau, gain):
    """Exact first-order line state x' = (gain u - x)/tau at the command nodes k dt (u piecewise linear)."""
    u = np.asarray(u, float); x = np.zeros(len(u))
    for k in range(len(u)-1):
        sl = (u[k+1]-u[k])/dt
        x[k+1] = gain*u[k+1] if tau == 0 else gain*(u[k]+sl*dt)-gain*sl*tau+(x[k]-gain*u[k]+gain*sl*tau)*np.exp(-dt/tau)
    return x


def _sampled_eval(u, x, dt, tau, gain, t):
    """Vectorized line response at local times t (array), from node states x; the command is zero after the last node."""
    u = np.asarray(u, float); t = np.atleast_1d(np.asarray(t, float)); n = len(u)-1
    out = np.zeros_like(t)
    inside = (t > 0) & (t < n*dt)
    if inside.any():
        k = np.minimum((t[inside]//dt).astype(int), n-1); h = t[inside]-k*dt; sl = (u[k+1]-u[k])/dt
        out[inside] = gain*(u[k]+sl*h) if tau == 0 else gain*(u[k]+sl*h)-gain*sl*tau+(x[k]-gain*u[k]+gain*sl*tau)*np.exp(-h/tau)
    after = t >= n*dt
    if after.any(): out[after] = 0. if tau == 0 else x[-1]*np.exp(-(t[after]-n*dt)/tau)
    return out


def _prepare_flux(model, s):
    """Per pulse: (coupler mode index, pulse, line, sampled-node states or None)."""
    labels = model.cell['mode_labels']; prep = []
    for f in s['flux']:
        line = model.cell['lines'][f['coupler']]
        nodes = _line_nodes(f['samples_Phi0'], f['dt_ns'], line['tau_ns'], line['gain']) if f['shape'] == 'sampled' else None
        prep.append((labels.index(f['coupler']), f, line, nodes))
    return prep


def _delivered_array(model, prep, i, t):
    t = np.atleast_1d(np.asarray(t, float)); phi = np.full_like(t, model.cell['idle_flux_Phi0'][i])
    for j, f, line, nodes in prep:
        if j != i: continue
        u = t-f['start_ns']; on = u >= 0
        if not on.any(): continue
        if nodes is None:
            phi[on] += delivered_flux(u[on], f['amplitude_Phi0'], f['duration_ns'], idle=0., tau_ns=line['tau_ns'], gain=line['gain'],
                                      settling=line['settling_time_constants'], rise_fraction=line['rise_fraction'])
        else:
            phi[on] += _sampled_eval(f['samples_Phi0'], nodes, f['dt_ns'], line['tau_ns'], line['gain'], u[on])
    return phi


def envelope_IQ(d, t):
    """(I, Q) of one validated microwave pulse at local schedule time t."""
    u = t-d['start_ns']; T = d['duration_ns']
    if u < 0 or u > T: return 0., 0.
    e = d['envelope']
    if e == 'square': return 1., 0.
    if e == 'cosine': return float(np.sin(np.pi*u/T)**2), 0.
    if e == 'flattop':
        r = d['ramp_ns']
        if u < r: return float(np.sin(.5*np.pi*u/r)**2), 0.
        if u > T-r: return float(np.sin(.5*np.pi*(T-u)/r)**2), 0.
        return 1., 0.
    if e in ('gaussian', 'drag'):
        c = T/2; sg = d['sigma_ns']; e0 = np.exp(-c*c/(2*sg*sg)); g = np.exp(-(u-c)**2/(2*sg*sg))
        I = (g-e0)/(1-e0)
        if e == 'gaussian': return float(I), 0.
        return float(I), float(d['drag_coefficient_ns']*(-(u-c)/(sg*sg))*g/(1-e0))
    n = len(d['samples_I'])-1; x = u/T*n; k = min(int(x), n-1); w = x-k
    return (float((1-w)*d['samples_I'][k]+w*d['samples_I'][k+1]), float((1-w)*d['samples_Q'][k]+w*d['samples_Q'][k+1]))


def controls_at_time_v2(model, s, t, t0=0., prep=None):
    prep = _prepare_flux(model, s) if prep is None else prep
    phi = {i: float(_delivered_array(model, prep, i, t)[0]) for i in model.coupler_modes}
    labels = model.cell['mode_labels']; mw = {}
    for d in s['drives']:
        I, Q = envelope_IQ(d, t)
        if I or Q:
            i = labels.index(d['mode']); arg = 2*np.pi*d['frequency_GHz']*(t+t0)+d['phase_rad']
            mw[i] = mw.get(i, 0.)+d['amplitude_GHz']*(I*np.cos(arg)+Q*np.sin(arg))
    return phi, mw


def _sampled_knot_indices(*channels):
    """Indices where at least one uniformly sampled channel changes slope."""
    differences = np.diff(np.asarray(channels, dtype=float), axis=1)
    return 1 + np.flatnonzero(np.any(differences[:, 1:] != differences[:, :-1], axis=0))


def integration_boundaries(schedule):
    """Return pulse edges and the corners of sampled controls in schedule time.

    The same boundaries are used by the integrator and its work estimate.
    A step crossing several corners could miss the entire nonzero part of
    a narrow waveform. Flat padding and equal adjacent slopes need no restart.
    """
    duration = float(schedule['duration_ns'])
    boundaries = [[0., duration]]
    for drive in schedule.get('drives', []):
        start = float(drive.get('start_ns', 0.))
        length = float(drive.get('duration_ns', duration-start))
        end = start+length
        boundaries.append([start, end])
        if drive.get('envelope') == 'flattop':
            ramp = float(drive.get('ramp_ns', 20.))
            boundaries.append([start+ramp, end-ramp])
        if drive.get('envelope') == 'sampled':
            in_phase = drive['samples_I']
            channels = (in_phase, drive['samples_Q']) if 'samples_Q' in drive else (in_phase,)
            knots = _sampled_knot_indices(*channels)
            boundaries.append(start+length*(knots/(len(in_phase)-1)))
    for pulse in schedule.get('flux', []):
        start = float(pulse.get('start_ns', 0.))
        boundaries.append([start])
        if pulse.get('shape') == 'sampled':
            values = pulse['samples_Phi0']
            spacing = float(pulse['dt_ns'])
            knots = _sampled_knot_indices(values)
            boundaries.append(start+spacing*np.r_[knots, len(values)-1])
    edges = np.concatenate(boundaries)
    return np.unique(edges[(edges >= 0.) & (edges <= duration)])


def propagate_states(model, schedule, V, *, rtol=1e-10, atol=1e-12, samples=2, keep_states=False, carrier_time_offset_ns=0.):
    """Evolve the columns of V (D x r, retained product basis) through a v2 schedule.

    Returns final (D x r), times, per-sample norms, and evidence. Flux-only schedules conserve the local-index parity
    exactly, so each column's even and odd parts are integrated as separate blocks (inherited algorithm). Any
    microwave breaks parity, which selects full-space DOP853 with a carrier-resolving maximum step. Both paths
    restart at pulse edges and sampled-control knots.
    """
    s = validate_schedule_v2(model, schedule)
    V = np.ascontiguousarray(np.asarray(V, complex))
    if V.ndim != 2 or V.shape[0] != model.basis.dimension: raise ValueError('V must be D x r in the retained basis')
    T = s['duration_ns']; r = V.shape[1]; t0 = _real(carrier_time_offset_ns, 'carrier offset', 0., 1e9)
    started = time.perf_counter()
    if T == 0 or r == 0:
        return dict(final=V.copy(), times_ns=np.array([0., T]), nfev=0, elapsed_s=0., engine='no evolution', schedule=s,
                    schedule_sha256=digest(s), norm_error=0., rtol=rtol, atol=atol)
    times = np.linspace(0, T, max(2, samples))
    flux_couplers = sorted({model.cell['mode_labels'].index(f['coupler']) for f in s['flux']})
    C = {i: model.cos_ops[i]-model.local[i][2][0, 0]*_eye(model) for i in flux_couplers}
    EJ = model.EJ; EJmax = {i: model.EJmax(i) for i in model.coupler_modes}

    prep = _prepare_flux(model, s)

    def deltas(t):
        phi, mw = controls_at_time_v2(model, s, t, t0, prep)
        return {i: float(squid_EJ_GHz(EJmax[i], phi[i])-EJ[i]) for i in flux_couplers}, mw

    def rayleigh(Hb, X):
        nrm = np.sum(np.abs(X)**2, axis=0)
        num = np.real(np.sum(X.conj()*_real_spmm(Hb, X), axis=0))
        return np.where(nrm > 0, num/np.where(nrm > 0, nrm, 1.), 0.)

    edges = integration_boundaries(s)
    Y = np.zeros((len(times), *V.shape), complex)
    nfev = 0
    if not s['drives']:
        blocks = []
        for p in (0, 1):
            rows = np.flatnonzero(model.basis.parity == p)
            X = V[rows]; cols = np.flatnonzero(np.sum(np.abs(X)**2, axis=0) > 0)
            if not len(cols): continue
            B = model.H[rows][:, rows].tocsr(); Cb = {i: C[i][rows][:, rows].tocsr() for i in flux_couplers}
            X = np.ascontiguousarray(X[:, cols]); E = rayleigh(B, X)
            blocks.append([rows, cols, B, Cb, E, X])
        for rows, cols, B, Cb, E, X in blocks:
            def rhs(t, y, B=B, Cb=Cb, E=E, shape=X.shape):
                dl, _ = deltas(t)
                Xt = y.reshape(shape); Z = _real_spmm(B, Xt)
                for i, dv in dl.items():
                    if dv: Z -= dv*_real_spmm(Cb[i], Xt)
                Z -= Xt*E
                return (-2j*np.pi*Z).ravel()
            y = X.ravel().copy(); out = np.empty((len(times), *X.shape), complex); out[0] = X
            for lo, hi in zip(edges[:-1], edges[1:]):
                idx = np.flatnonzero((times > lo) & (times <= hi)); req = times[idx]
                ev = req if len(req) and req[-1] == hi else np.r_[req, hi]
                sol = solve_ivp(rhs, (lo, hi), y, method='DOP853', rtol=rtol, atol=atol, t_eval=ev)
                if not sol.success: raise RuntimeError(sol.message)
                out[idx] = sol.y[:, :len(idx)].T.reshape(len(idx), *X.shape); y = sol.y[:, -1]; nfev += int(sol.nfev)
            out *= np.exp(-2j*np.pi*times[:, None, None]*E[None, None, :])
            for j, c in enumerate(cols): Y[:, rows, c] = out[:, :, j]
        engine = 'parity-block flux-only DOP853 on arbitrary vectors (v2 schedule)'
    else:
        H = model.H; N = model.charge_ops; E = rayleigh(H, V); shape = V.shape

        def rhs(t, y):
            dl, mw = deltas(t)
            X = y.reshape(shape); Z = _real_spmm(H, X)
            for i, dv in dl.items():
                if dv: Z -= dv*_real_spmm(C[i], X)
            for i, a in mw.items():
                if a: Z += a*_real_spmm(N[i], X)
            Z -= X*E
            return (-2j*np.pi*Z).ravel()
        carrier = max(d['frequency_GHz'] for d in s['drives'])
        max_step = min(T, 1/(CARRIER_STEPS_PER_CYCLE*carrier)) if carrier > 0 else T
        Y[0] = V; y = V.ravel().copy()
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = np.flatnonzero((times > lo) & (times <= hi)); req = times[idx]
            ev = req if len(req) and req[-1] == hi else np.r_[req, hi]
            sol = solve_ivp(rhs, (lo, hi), y, method='DOP853', rtol=rtol, atol=atol, max_step=max_step, t_eval=ev)
            if not sol.success: raise RuntimeError(sol.message)
            Y[idx] = sol.y[:, :len(idx)].T.reshape(len(idx), *shape); y = sol.y[:, -1]; nfev += int(sol.nfev)
        Y *= np.exp(-2j*np.pi*times[:, None, None]*E[None, None, :])
        engine = 'full-space parity-breaking lab-frame DOP853 on arbitrary vectors; no RWA (v2 schedule)'
    g0 = V.conj().T@V
    norm_err = max(float(np.abs(x.conj().T@x-g0).max()) for x in Y)
    out = dict(final=Y[-1].copy(), times_ns=times, nfev=nfev, elapsed_s=time.perf_counter()-started, engine=engine, schedule=s,
               schedule_sha256=digest(s), norm_error=norm_err, rtol=rtol, atol=atol, carrier_time_offset_ns=t0)
    if keep_states: out['states'] = Y
    return out
