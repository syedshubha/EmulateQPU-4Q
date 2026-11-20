"""Measurement instruments for the chain (0.1.0.dev4), generalized from the two-qubit twin.

Source model: emulateqpu_twoq.measurement (twoq_measurement.v1). The physics and conventions are the twin's; this
module only generalizes the bookkeeping:
- any subset of data qubits, one readout channel per measured qubit;
- optional declared cross-dispersive shifts of other data qubits on a channel;
- states carried as branch ensembles rho = V W V^dag rather than full density matrices, so the instrument scales to
  the capped chain bases;
- a discrete classified-heterodyne outcome (declared threshold classifier, integrated in closed form) for feedback.

Instruments (all exact for the declared model, nothing postselected):

1. ideal_projective
   - local_levels: projectors on the bare local levels of the measured data modes (the twin's local_levels basis).
     Outcomes are bitstrings over the measured qubits plus 'leakage' (any measured level >= 2), or every level
     tuple when resolution = 'all_levels'.
   - dressed_computational: the 2^n dressed computational vectors plus their complement 'leakage'. Allowed only when
     every data qubit is measured; a subset of a multi-qubit dressed basis is not a local measurement.
2. dispersive_qnd_heterodyne (twin model). Each channel is a readout resonator coupled to a filter, with the
   terminal vacuum Markov channel L = sqrt(kappa_f) a_f. During the window, the bare local levels of the affected
   data modes are frozen. The response of each level label gives observed heterodyne amplitudes (efficiency eta),
   hidden amplitudes (unmeasured temporal modes, lost output, residual cavity fields) and a drive-induced phase.
   The map is the Schur product rho -> rho o K over level labels, applied in the retained product basis:
   - unconditional: K = hidden kernel o observed-overlap kernel (the twin's unconditional_kernel);
   - record: the twin's condition_coherent for one heterodyne record z (given or sampled exactly as the twin does);
   - classified: per channel, the declared half-plane classifier Re(e^{-i phi}(z - m)) > 0. Default: the
     perpendicular bisector of the channel qubit's |0> and |1> centroids, with all other affected levels 0. The
     half-plane integral of <z|a><b|z> d^2z/pi is (1/2) exp(a b* - (|a|^2+|b|^2)/2) erfc(c - (a' + b'*)/2) in the
     rotated frame, so the classified instrument is exact (no sampling), and the outcomes sum to the unconditional
     kernel.

Scope (as for the twin): an assumed effective readout model, not geometry-derived. It has no qubit relaxation, no
Purcell decay, no measurement-induced state transitions and no shared-bath claim. A kernel acts on bare local levels;
it is not QND with respect to the interacting idle Hamiltonian, and that non-commutation is physical content of
the model (it is where local-measurement energy injection comes from), not an error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np
from scipy.linalg import expm, eigh, qr
from scipy.special import erfc, erfcx, logsumexp

VERSION = 'chain_measurement.v1'
TWIN_SOURCE = 'emulateqpu_twoq.measurement twoq_measurement.v1'


def _replacement_state(spec, dims):
    """Validate a declared local or retained-space replacement, without renormalizing."""
    d = math.prod(dims)
    if spec is None:
        spec = {'levels': [0]*len(dims)}
    if isinstance(spec, dict) and set(spec) == {'levels'}:
        levels = spec['levels']
        if (not isinstance(levels, list) or len(levels) != len(dims)
            or any(type(v) is not int or v < 0 or v >= n for v, n in zip(levels, dims))):
            raise ValueError('Replacement levels must fit the explicitly reset subsystem')
        psi = np.zeros(d, complex); psi[np.ravel_multi_index(tuple(levels), dims)] = 1.
        return psi[:, None], np.ones(1)
    if isinstance(spec, dict):
        if set(spec)-{'real', 'imag'} or 'real' not in spec:
            raise ValueError('Replacement state requires levels or real/imag arrays')
        a = np.asarray(spec['real'], float)+1j*np.asarray(spec.get('imag', np.zeros_like(spec['real'])), float)
    else:
        a = np.asarray(spec, complex)
    if not np.isfinite(a).all():
        raise ValueError('Replacement state must be finite')
    if a.shape == (d,):
        if abs(np.vdot(a, a)-1.) > 1e-12:
            raise ValueError('Replacement vector must have unit norm')
        return a[:, None], np.ones(1)
    if a.shape != (d, d) or np.max(abs(a-a.conj().T)) > 1e-12 or abs(np.trace(a)-1.) > 1e-12:
        raise ValueError('Replacement density must be Hermitian with unit trace')
    vals, vecs = eigh(a)
    if vals.min() < -1e-12:
        raise ValueError('Replacement density must be positive semidefinite')
    # Negative eigensolver roundoff is explicitly excluded, without normalization.
    keep = vals > 0.
    return vecs[:, keep], vals[keep]


def replacement_reset(model, branch, *, modes=None, state=None, kind='local', max_output_bytes=256*1024**2):
    """Trace-preserving replacement in the stated retained space.

    local: rho -> sigma_modes tensor Tr_modes(rho), in original subsystem order.
    full: X -> Tr(X) sigma, where sigma is an explicit retained-basis state.
    No cap projection or renormalization is performed. Local maps are rejected
    unless every output of the map is closed in the retained capped basis.
    Correlations with the replaced subsystem are removed. Process branches use
    the same Kraus map on all input columns and retain their coherences.
    """
    if kind not in ('local', 'full'):
        raise ValueError('Replacement kind must be local or full')
    D = model.basis.dimension
    labels = model.cell['mode_labels']; S = model.basis.states; dims = model.basis.dims
    if kind == 'full':
        if modes is not None:
            raise ValueError('Full replacement has no subsystem selection')
        vecs, vals = _replacement_state(state, (D,))
        if branch.block == 0:
            mass = branch.probability()
            return branch.copy(V=vecs.copy(), W=np.diag(vals*mass), log=branch.log+[
                dict(operation='fixed_state_replacement', input_trace=mass,
                     removes_all_correlations=True, cap_projection=False)])
        count = D*len(vals)
    else:
        if (not isinstance(modes, (list, tuple)) or not modes or len(set(modes)) != len(modes)
            or any(m not in labels for m in modes)):
            raise ValueError('Local replacement requires unique mode labels')
        indices = tuple(labels.index(m) for m in modes)
        local_dims = tuple(dims[i] for i in indices)
        vecs, vals = _replacement_state(state, local_dims)
        local_tuples = np.array(list(np.ndindex(local_dims)), dtype=int)
        old_labels = np.ravel_multi_index(S[:, indices].T, local_dims)
        lookup = {tuple(row): j for j, row in enumerate(S.tolist())}
        support = np.flatnonzero(np.any(abs(vecs) > 0., axis=1))
        mappings = {}
        for target in support:
            out_rows = S.copy()
            out_rows[:, indices] = local_tuples[target]
            mapped = np.array([lookup.get(tuple(row), -1) for row in out_rows.tolist()])
            if np.any(mapped < 0):
                raise ValueError('Replacement is not closed in the excitation-capped space; use an explicitly expanded model')
            mappings[int(target)] = mapped
        count = math.prod(local_dims)*len(vals)
    r = branch.V.shape[1]
    estimate = 16*D*r*count+16*(r*count)**2
    if estimate > max_output_bytes:
        raise ValueError('Replacement branch factors exceed the declared output memory limit before allocation')
    out = []
    if kind == 'full':
        groups = ((j, vecs[:, k], float(p)) for k, p in enumerate(vals) for j in range(D))
        for j, psi, p in groups:
            out.append(np.sqrt(p)*psi[:, None]*branch.V[j, :][None, :])
    else:
        groups = ((old, k) for old in range(math.prod(local_dims)) for k in range(len(vals)))
        for old, k in groups:
            columns = np.zeros_like(branch.V)
            src = np.flatnonzero(old_labels == old)
            for target in support:
                np.add.at(columns, mappings[int(target)][src], np.sqrt(vals[k])*vecs[target, k]*branch.V[src])
            out.append(columns)
    V = np.hstack(out)
    W = np.kron(np.eye(count), branch.W)
    reset = branch.copy(V=V, W=W, log=branch.log+[
        dict(operation='local_replacement' if kind == 'local' else 'fixed_state_replacement',
             modes=list(modes) if modes is not None else None, input_trace=branch.probability(),
             removes_replaced_correlations=True, cap_projection=False)])
    if abs(reset.probability()-branch.probability()) > 1e-10*max(1., branch.probability()):
        raise ValueError('Replacement trace preservation check failed')
    return reset


# ------------------------------------------------------------------ branch ensemble
@dataclass
class Branch:
    """One reported-record branch: subnormalized rho = V W V^dag; W Hermitian PSD."""
    record: dict
    V: np.ndarray
    W: np.ndarray
    frame: np.ndarray                     # software (virtual-Z) frame angle per data qubit, metrics convention
    discarded_trace: float = 0.
    log: list = field(default_factory=list)
    block: int = 0                        # process mode: columns form Kraus blocks of `block` labelled inputs, W = I

    def probability(self):
        return float(np.real(np.trace(self.W@(self.V.conj().T@self.V))))

    def diagonal(self):
        return np.real(np.einsum('sj,jk,sk->s', self.V, self.W, self.V.conj()))

    def expectation(self, O):
        """Tr(O rho) for a sparse or dense D x D operator (unnormalized)."""
        return complex(np.trace(self.W@(self.V.conj().T@(O@self.V))))

    def density(self):
        return self.V@self.W@self.V.conj().T

    def copy(self, **changes):
        b = Branch(dict(self.record), self.V.copy(), self.W.copy(), self.frame.copy(), self.discarded_trace, list(self.log), self.block)
        for k, v in changes.items(): setattr(b, k, v)
        return b


def pure_branch(psi, n_data, record=None):
    psi = np.asarray(psi, complex).reshape(-1, 1)
    return Branch(dict(record or {}), psi, np.ones((1, 1), complex), np.zeros(n_data))


def process_branch(V, n_data, record=None):
    """Process mode: columns are labelled input states propagated together; the channel on |v><v'| is
    sum_j B_j[:, v] B_j[:, v']^dag over the Kraus blocks. probability() is the summed trace over inputs."""
    V = np.asarray(V, complex)
    return Branch(dict(record or {}), V.copy(), np.eye(V.shape[1], dtype=complex), np.zeros(n_data), block=V.shape[1])


def process_blocks(branch):
    """Process mode: the J Kraus blocks (each D x block) of the branch."""
    r = branch.block; J = branch.V.shape[1]//r
    return [branch.V[:, j*r:(j+1)*r] for j in range(J)]


def compress(branch, tol=1e-14):
    """Exact re-factorization rho = Q (R W R^dag) Q^dag; drops eigenvalues <= tol * trace and records them.

    Process mode (branch.block = r > 0): the input-label structure must survive, so only the Kraus index is
    compressed. The blocks B_j are rows vec(B_j) of X, and X = U S Vh gives new blocks s_k Vh_k. This is exact for kept
    singular values, because sum_j B_j (.) B_j^dag is invariant under unitary recombination of j."""
    V, W = branch.V, branch.W
    if V.shape[1] == 0: return branch
    if branch.block:
        r = branch.block; D = V.shape[0]; J = V.shape[1]//r
        X = np.stack([V[:, j*r:(j+1)*r].ravel() for j in range(J)])
        U, sv, Vh = np.linalg.svd(X, full_matrices=False)
        tr = float(np.sum(sv**2)); keep = sv**2 > tol*max(tr, 1e-300)
        blocks = [(sv[k]*Vh[k]).reshape(D, r) for k in np.flatnonzero(keep)]
        b = branch.copy(V=np.hstack(blocks) if blocks else np.zeros((D, 0), complex), W=np.eye(r*len(blocks), dtype=complex))
        # Match probability(): summed trace over the labelled input columns.
        # A per-input mean can be obtained by dividing by r explicitly.
        b.discarded_trace += float(np.sum(sv[~keep]**2))
        return b
    Q, R = qr(V, mode='economic')
    M = R@W@R.conj().T; M = (M+M.conj().T)/2
    lam, U = eigh(M)
    tr = float(max(lam.sum(), 0.))
    if lam.min() < -1e-10*max(tr, 1e-300): raise ValueError('Branch weight is not positive semidefinite beyond roundoff')
    keep = lam > tol*max(tr, 1e-300)
    dropped = float(np.sum(np.abs(lam[~keep])))
    b = branch.copy(V=np.ascontiguousarray(Q@U[:, keep]), W=np.diag(lam[keep]).astype(complex))
    b.discarded_trace += dropped
    return b


def _apply_diag_kraus(branch, G, lab):
    """rho -> sum_j D_j rho D_j^dag, D_j = diag(G[lab, j]): V -> [D_1 V, ..., D_J V], W -> I_J (x) W."""
    J = G.shape[1]
    if J == 0: return branch.copy(V=np.zeros((branch.V.shape[0], 0), complex), W=np.zeros((0, 0), complex))
    Vs = [G[lab, j][:, None]*branch.V for j in range(J)]
    return branch.copy(V=np.hstack(Vs), W=np.kron(np.eye(J), branch.W))


def kernel_factor(K, tol=1e-15):
    """K (L x L Hermitian PSD) = G G^dag. Returns G and the dropped spectral weight (sum |lambda| <= tol * max)."""
    K = (K+K.conj().T)/2
    lam, U = eigh(K)
    top = max(float(lam.max()), 1e-300)
    if lam.min() < -1e-10*top: raise ValueError('Instrument kernel is not positive semidefinite (min eigenvalue %.3g)' % lam.min())
    keep = lam > tol*top
    return U[:, keep]*np.sqrt(lam[keep])[None, :], float(np.sum(np.abs(lam[~keep])))


# ------------------------------------------------------------------ helpers
def _num(x, name, lo=None, hi=None):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating)):
        raise ValueError(name+' must be numeric, not bool')
    y = float(x)
    if not math.isfinite(y) or (lo is not None and y < lo) or (hi is not None and y > hi):
        raise ValueError(name+' is nonfinite or outside its declared range')
    return y


def _cnum(x, name, hi):
    if isinstance(x, (list, tuple)) and len(x) == 2: z = complex(_num(x[0], name), _num(x[1], name))
    elif isinstance(x, dict) and set(x) == {'real', 'imag'}: z = complex(_num(x['real'], name), _num(x['imag'], name))
    elif isinstance(x, complex): z = x
    else: z = complex(_num(x, name))
    if abs(z) > hi: raise ValueError(name+' exceeds its bound')
    return z


def _digest(md, arrays):
    h = hashlib.sha256(json.dumps(md, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    for k, v in sorted(arrays.items()):
        a = np.ascontiguousarray(v); h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _coherent_overlap(a):
    """K[s,t] = <a_t|a_s> (twin convention, density-matrix multiplier)."""
    a = np.asarray(a, complex)
    norm = np.sum(np.abs(a)**2, axis=1)
    return np.exp(a@a.conj().T-.5*(norm[:, None]+norm[None, :]))


def data_index(model, qubit):
    labels = model.cell['mode_labels']
    if qubit not in labels or labels.index(qubit) not in model.data_modes: raise ValueError('Unknown data qubit '+str(qubit))
    return labels.index(qubit)


def level_labels(model, modes):
    """Per retained basis state: flat index of its level tuple on `modes`; the tuple table."""
    dims = [model.basis.dims[i] for i in modes]
    lab = np.ravel_multi_index(model.basis.states[:, list(modes)].T.astype(np.int64), dims) if modes else np.zeros(model.basis.dimension, np.int64)
    return lab, np.array(list(np.ndindex(*dims))) if modes else np.zeros((1, 0), int)


# ------------------------------------------------------------------ ideal projective
def ideal_outcomes(model, qubits, basis='local_levels', resolution='computational_plus_leakage'):
    """List of (label, kind, data). kind 'mask': diagonal 0/1 over basis rows; kind 'vec': dressed projector columns."""
    modes = [data_index(model, q) for q in qubits]
    if len(set(modes)) != len(modes) or not modes: raise ValueError('Nonempty unique measured qubit list required')
    if basis == 'local_levels':
        lev = model.basis.states[:, modes]
        out = []
        if resolution == 'computational_plus_leakage':
            for bits in np.ndindex((2,)*len(modes)):
                out.append((''.join(map(str, bits)), 'mask', np.all(lev == np.array(bits), axis=1)))
            out.append(('leakage', 'mask', np.any(lev >= 2, axis=1)))
        elif resolution == 'all_levels':
            for t in np.ndindex(*[model.basis.dims[i] for i in modes]):
                m = np.all(lev == np.array(t), axis=1)
                if m.any(): out.append((''.join(map(str, t)), 'mask', m))
        else: raise ValueError('resolution must be computational_plus_leakage or all_levels')
        return out
    if basis == 'dressed_computational':
        if sorted(modes) != sorted(model.data_modes): raise ValueError('dressed_computational requires measuring every data qubit')
        if model.analysis.get('encoding_kind', 'dressed') != 'dressed' or model.encoding.shape[1] != 2**len(model.data_modes):
            raise ValueError('dressed_computational measurement requires the complete dressed encoding; use local_levels for selected inputs')
        nd = len(model.data_modes); order = [model.data_modes.index(i) for i in modes]
        labels = model.analysis.get('encoding_labels') or [''.join(map(str, b)) for b in np.ndindex((2,)*nd)]
        out = []
        for k, label in enumerate(labels):
            bits = tuple(map(int, label))
            out.append((''.join(str(bits[o]) for o in order), 'vec', model.encoding[:, k:k+1]))
        out.append(('leakage', 'complement', model.encoding))
        return out
    raise ValueError('basis must be local_levels or dressed_computational')


def apply_ideal(branch, outcomes):
    res = []
    for label, kind, data in outcomes:
        if kind == 'mask': V = branch.V*data[:, None]
        elif kind == 'vec': V = data@(data.conj().T@branch.V)
        else: V = branch.V-data@(data.conj().T@branch.V)
        res.append((label, branch.copy(V=V)))
    return res


# ------------------------------------------------------------------ dispersive QND heterodyne
QND_KEYS = {'model', 'basis', 'duration_ns', 'drive_duration_ns', 'response_order', 'max_photons', 'channels'}
CHANNEL_KEYS = {'name', 'reads', 'readout_frequency_GHz', 'filter_frequency_GHz', 'drive_frequency_GHz', 'coupling_MHz',
                'filter_kappa_per_ns', 'drive_amplitude_sqrt_photons_per_ns', 'efficiency', 'chi_MHz', 'level_shifts_MHz'}


def qnd_defaults(qubits):
    """The twin's declared defaults, one channel per measured qubit (readout 7.2 + 0.4 k GHz). Not device values."""
    return dict(model='dispersive_qnd_heterodyne', basis='local_levels', duration_ns=300., drive_duration_ns=200., response_order=32,
                max_photons=1., channels=[dict(name='r_'+q, reads=q, readout_frequency_GHz=7.2+.4*k, coupling_MHz=40.,
                                               filter_kappa_per_ns=.03, drive_amplitude_sqrt_photons_per_ns=.01, efficiency=.5,
                                               chi_MHz={q: 1.}) for k, q in enumerate(qubits)])


@dataclass(frozen=True)
class QNDInstrument:
    modes: tuple                # affected data mode indices (level labels)
    levels: np.ndarray          # (L, len(modes)) level tuples
    observed: np.ndarray        # (L, M) observed heterodyne amplitudes
    hidden: np.ndarray          # (L, Hn) hidden amplitudes
    phases: np.ndarray          # (L,) drive-induced phases
    hidden_kernel: np.ndarray
    unconditional_kernel: np.ndarray
    channel_qubit_modes: tuple  # mode index read by each channel
    metadata: dict
    instrument_id: str


def qnd_instrument(model, config):
    """Build the twin's coherent instrument over the level labels of the affected data modes."""
    c = dict(config)
    if set(c)-QND_KEYS: raise ValueError('Unsupported QND keys: '+str(sorted(set(c)-QND_KEYS)))
    if c.get('basis', 'local_levels') != 'local_levels': raise ValueError('QND response labels bare local levels (twin basis)')
    T = _num(c.get('duration_ns', 300.), 'duration_ns', .1, 2000.)
    Td = _num(c.get('drive_duration_ns', min(200., T)), 'drive_duration_ns', 0., T)
    order = c.get('response_order', 32)
    if type(order) is not int or not 4 <= order <= 128: raise ValueError('response_order integer 4..128 required')
    photon_limit = _num(c.get('max_photons', 1.), 'max_photons', 0., 10.)
    chans = c.get('channels')
    if not isinstance(chans, list) or not 1 <= len(chans) <= 9: raise ValueError('1..9 readout channels required')
    parsed = []
    for k, ch in enumerate(chans):
        if not isinstance(ch, dict) or set(ch)-CHANNEL_KEYS: raise ValueError('Unsupported channel keys')
        q = ch.get('reads'); qi = data_index(model, q)
        fr = _num(ch.get('readout_frequency_GHz', 7.2+.4*k), 'readout frequency', 1., 20.)
        p = dict(name=str(ch.get('name', 'r_'+q)), reads=q, mode=qi, fr=fr,
                 ff=_num(ch.get('filter_frequency_GHz', fr), 'filter frequency', 1., 20.),
                 fd=_num(ch.get('drive_frequency_GHz', fr), 'drive frequency', 1., 20.),
                 J=_num(ch.get('coupling_MHz', 40.), 'coupling', 0., 500.)*2*np.pi*1e-3,
                 kappa=_num(ch.get('filter_kappa_per_ns', .03), 'filter kappa', 0., 1.),
                 beta=_cnum(ch.get('drive_amplitude_sqrt_photons_per_ns', .01), 'drive amplitude', .2),
                 eta=_num(ch.get('efficiency', .5), 'efficiency', 0., 1.))
        if 'level_shifts_MHz' in ch:
            if 'chi_MHz' in ch: raise ValueError('Choose explicit level shifts or the declared linear chi extension')
            sh = {}
            for qq, v in ch['level_shifts_MHz'].items():
                m = data_index(model, qq); a = np.asarray(v, float)
                if a.shape != (model.basis.dims[m],) or not np.isfinite(a).all() or np.max(np.abs(a)) > 1000:
                    raise ValueError('level_shifts_MHz needs one bounded shift per retained level of '+qq)
                sh[m] = a
            p['shift_scope'] = 'explicit assumed per-level shifts'
        else:
            chi = ch.get('chi_MHz', {q: 1.}); sh = {}
            for qq, v in chi.items():
                m = data_index(model, qq); x = _num(v, 'chi', -100., 100.)
                sh[m] = x*(2*np.arange(model.basis.dims[m])-1)
            p['shift_scope'] = 'assumed linear higher-level extension chi*(2j-1) (twin default)'
        p['shifts'] = {m: a for m, a in sh.items() if np.any(a != 0)}
        parsed.append(p)
    active = [p for p in parsed if p['beta'] != 0]
    modes = tuple(sorted({m for p in active for m in p['shifts']} | {p['mode'] for p in parsed}))
    lab, levels = level_labels(model, modes)
    present = np.unique(lab)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    ts, ws = [], []
    for lo, hi in [(0., Td), (Td, T)]:
        if hi > lo: ts.extend(lo+(nodes+1)*(hi-lo)/2); ws.extend(weights*(hi-lo)/2)
    ts, ws = np.asarray(ts), np.asarray(ws)
    if abs(ws.sum()-T) > 1e-10*max(1., T): raise ValueError('Response quadrature does not span the window')
    mode_fn = np.sqrt(ws/ws.sum())
    L = len(levels); M = len(parsed)
    observed = np.zeros((L, M), complex); phases = np.zeros(L); hidden_parts = []
    peak = 0.; residual_max = 0.
    for ci, p in enumerate(parsed):
        shift = np.zeros(L)
        for m, a in p['shifts'].items(): shift += a[levels[:, modes.index(m)]]
        output = np.zeros((L, len(ts)), complex); final = np.zeros((L, 2), complex)
        cache = {}
        for s in present:
            key = float(shift[s])
            if key not in cache:
                omega = np.array([[2*np.pi*(p['fr']-p['fd']+key*1e-3), p['J']], [p['J'], 2*np.pi*(p['ff']-p['fd'])]], float)
                loss = np.array([0., np.sqrt(p['kappa'])])
                A = -1j*omega-.5*np.outer(loss, loss); force = -loss*p['beta']
                aug = np.zeros((3, 3), complex); aug[:2, :2] = A; aug[:2, 2] = force
                at_switch = expm(aug*Td)[:2, 2]
                out = np.zeros(len(ts), complex); pk = 0.
                for k, t in enumerate(ts):
                    alpha = expm(aug*t)[:2, 2] if t <= Td else expm(A*(t-Td))@at_switch
                    out[k] = loss@alpha; pk = max(pk, float(np.sum(np.abs(alpha)**2)))
                    rhs = A@alpha+(force if t < Td else 0.); incoming = p['beta'] if t < Td else 0.
                    res = 2*np.vdot(alpha, rhs).real-(abs(incoming)**2-abs(incoming+loss@alpha)**2)
                    residual_max = max(residual_max, abs(float(res)))
                fin = expm(A*(T-Td))@at_switch; pk = max(pk, float(np.sum(np.abs(fin)**2)))
                ph = float(np.sum(ws*np.where(ts < Td, np.imag(np.conj(p['beta'])*out), 0.)))
                cache[key] = (out, fin, pk, ph)
            out, fin, pk, ph = cache[key]
            output[s] = out; final[s] = fin; peak = max(peak, pk); phases[s] += ph
        emission = output*np.sqrt(ws)[None, :]; projection = emission@mode_fn
        observed[:, ci] = np.sqrt(p['eta'])*projection
        hidden_parts.extend([emission-projection[:, None]*mode_fn[None, :], np.sqrt(1-p['eta'])*projection[:, None], final])
    if peak > photon_limit: raise ValueError(f'Sampled photon occupation {peak:g} exceeds the scenario limit {photon_limit:g}')
    hidden = np.concatenate(hidden_parts, axis=1)
    hk = _coherent_overlap(hidden)*np.exp(1j*(phases[:, None]-phases[None, :]))
    uk = hk*_coherent_overlap(observed)
    md = dict(version=VERSION, source=TWIN_SOURCE, model='dispersive_qnd_heterodyne', basis='local_levels',
              affected_modes=[model.cell['mode_labels'][m] for m in modes], duration_ns=T, drive_duration_ns=Td, response_order=order,
              channels=[dict(name=p['name'], reads=p['reads'], readout_frequency_GHz=p['fr'], filter_frequency_GHz=p['ff'],
                             drive_frequency_GHz=p['fd'], coupling_MHz=p['J']/(2*np.pi*1e-3), filter_kappa_per_ns=p['kappa'],
                             beta=[p['beta'].real, p['beta'].imag], efficiency=p['eta'], shift_scope=p['shift_scope'],
                             level_shifts_MHz={model.cell['mode_labels'][m]: a.tolist() for m, a in p['shifts'].items()}) for p in parsed],
              receiver='one orthogonal output band per channel; normalized rectangular heterodyne; input background subtracted',
              IQ_vacuum_variance_per_quadrature=.5, max_sampled_chain_photons=peak, scenario_photon_limit=photon_limit,
              passive_flux_balance_max_abs_per_ns=residual_max,
              approximations=['Frozen bare local levels of every mode during the window (twin)',
                              'Linear passive resonator/filter chains with vacuum Markov terminal baths (twin)',
                              'No qubit relaxation, Purcell decay, measurement-induced transitions or shared bath (twin)',
                              'Kernel applied in the bare product basis after the idle/concurrent evolution of the window (chain program convention)'],
              physical_port_qualified=False, dispersive_validity_qualified=False)
    arrays = dict(observed=observed, hidden=hidden, phases=phases)
    return QNDInstrument(modes, levels, observed, hidden, phases, hk, uk, tuple(p['mode'] for p in parsed), md, _digest(md, arrays)), lab


def _halfplane(a, b, c, sign):
    """F[s,t] = int_{sign*(x-c)>0} <z|a_s><a_t|z> d^2z/pi (rotated frame), exactly; a, b are 1-D complex arrays."""
    A = a[:, None]; B = b[None, :]
    base = A*B.conj()-.5*(np.abs(A)**2+np.abs(B)**2)
    w = c-(A+B.conj())/2
    if sign < 0: w = -w
    # (1/2) exp(base) erfc(w); erfcx where Re w >= 0 avoids overflow/underflow of the product
    pos = w.real >= 0
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        hi = .5*np.exp(base-w*w)*erfcx(np.where(pos, w, 0.))
        lo = .5*np.exp(base)*erfc(np.where(pos, 0., w))
    out = np.where(pos, hi, lo)
    if not np.isfinite(out).all(): raise ValueError('Classified kernel is not representable')
    return out


def classifier(inst, spec=None):
    """Per channel (angle phi, threshold c): outcome '1' iff Re(e^{-i phi} z) > c. Default: bisector of the channel
    qubit's 0/1 centroids with every other affected level 0."""
    out = []
    for ci, m in enumerate(inst.channel_qubit_modes):
        if spec is not None and spec.get(str(ci)) is not None:
            v = spec[str(ci)]; out.append((float(v['angle_rad']), float(v['threshold']))); continue
        pos = inst.modes.index(m)
        z = np.zeros(len(inst.modes), int); o = np.zeros(len(inst.modes), int); o[pos] = 1
        i0 = int(np.flatnonzero(np.all(inst.levels == z, axis=1))[0]); i1 = int(np.flatnonzero(np.all(inst.levels == o, axis=1))[0])
        a0, a1 = inst.observed[i0, ci], inst.observed[i1, ci]
        if abs(a1-a0) < 1e-12: raise ValueError('Channel %d centroids coincide; no bisector classifier exists' % ci)
        phi = float(np.angle(a1-a0)); thr = float(np.real(np.exp(-1j*phi)*(a0+a1)/2))
        out.append((phi, thr))
    return out


def classified_kernels(inst, cls):
    """{bitstring over channels: K_y}; sum_y K_y = unconditional kernel."""
    M = inst.observed.shape[1]; out = {}
    per = []
    for ci in range(M):
        phi, c = cls[ci]; a = np.exp(-1j*phi)*inst.observed[:, ci]
        per.append({'1': _halfplane(a, a, c, +1), '0': _halfplane(a, a, c, -1)})
    for bits in np.ndindex((2,)*M):
        K = inst.hidden_kernel.copy()
        for ci, b in enumerate(bits): K = K*per[ci][str(b)]
        out[''.join(map(str, bits))] = K
    return out


def sample_record(inst, lab, branch, seed):
    """One heterodyne record from the branch's distribution, sampled as the twin does (latent row, then noise)."""
    p = branch.diagonal(); p = np.maximum(p, 0.); p /= p.sum()
    rng = np.random.default_rng(seed); s = rng.choice(len(p), size=1, p=p)
    M = inst.observed.shape[1]
    return inst.observed[lab[s[0]]]+(rng.normal(size=(1, M))+1j*rng.normal(size=(1, M)))[0]/np.sqrt(2)


def record_factor(inst, z):
    """Diagonal record amplitude per label (twin's condition_coherent), common real rescaling, plus log-normalizer."""
    z = np.asarray(z, complex)
    d = np.sum(np.abs(z[None, :]-inst.observed)**2, axis=1); la = -.5*d
    return np.exp(la-la.max()+1j*np.imag(inst.observed@z.conj()))


def apply_qnd(branch, inst, lab, outcome='unconditional', *, classifier_spec=None, record=None, seed=0, tol=1e-15):
    """Returns [(label, branch)] and evidence. 'record' returns one normalized branch with the record's log-likelihood."""
    ev = dict(instrument_id=inst.instrument_id, outcome_mode=outcome)
    if outcome == 'unconditional':
        G, drop = kernel_factor(inst.unconditional_kernel, tol); ev['kernel_dropped_weight'] = drop
        return [('unrecorded', _apply_diag_kraus(branch, G, lab))], ev
    if outcome == 'classified':
        cls = classifier(inst, classifier_spec); ev['classifier'] = [dict(angle_rad=a, threshold=c) for a, c in cls]
        res = []; ev['kernel_dropped_weight'] = {}
        for y, K in classified_kernels(inst, cls).items():
            G, drop = kernel_factor(K, tol); ev['kernel_dropped_weight'][y] = drop
            res.append((y, _apply_diag_kraus(branch, G, lab)))
        return res, ev
    if outcome == 'record':
        if branch.block > 1:
            raise ValueError('A normalized continuous-record trajectory is not a multi-input channel; select one physical input')
        if isinstance(record, dict):
            if set(record) != {'real', 'imag'}:
                raise ValueError('Supplied I/Q record requires real and imag arrays')
            record = np.asarray(record['real'], float)+1j*np.asarray(record['imag'], float)
        z = sample_record(inst, lab, branch, seed) if record is None else np.asarray(record, complex)
        if z.shape != (inst.observed.shape[1],): raise ValueError('Record must have one complex value per channel')
        amp = record_factor(inst, z)
        Gh, drop = kernel_factor(inst.hidden_kernel, tol)
        b = _apply_diag_kraus(branch, amp[:, None]*Gh, lab)
        tr = b.probability(); p0 = branch.probability()
        if not tr > 0: raise ValueError('Record has no representable likelihood')
        b.V = b.V*np.sqrt(p0/tr)
        prior = branch.diagonal()/p0; d = np.sum(np.abs(z[None, :]-inst.observed)**2, axis=1)
        nz = prior > 0; M = len(z)
        ev.update(record=dict(real=z.real.tolist(), imag=z.imag.tolist()), seed=None if record is not None else seed,
                  log_likelihood=float(logsumexp(np.log(prior[nz])-d[lab[nz]])-M*np.log(np.pi)), kernel_dropped_weight=drop,
                  normalization='trajectory branch renormalized to the parent branch probability')
        return [('record', b)], ev
    raise ValueError('outcome must be unconditional, classified or record')
