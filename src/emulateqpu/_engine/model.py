"""Finite chain of fixed-frequency transmons and flux-tunable transmon couplers.

Units: H/h in GHz, time in ns, flux in Phi/Phi0, charge in Cooper pairs.
Each local mode is the charge-basis cosine Hamiltonian 4 EC_ii n^2 - EJ cos(phi)
at ng=0, diagonalized at its idle EJ and projected onto its lowest L states.
The local block is the exact retained eigenenergy diagonal (never 4EC(PnP)^2).
Every signed off-diagonal 8 EC_ij n_i n_j term of the supplied full EC matrix is
kept, including non-neighbour parasitics. No RWA, no Duffing model, no coupler
elimination.

An optional local-excitation cap N keeps product states with
sum(w_i * local index_i) <= N, with weights w_i = 1 by default. Modes listed in
cap_exempt get w_i = 0 and keep every retained level for every retained tuple of
the other modes (0.1.0.dev2: needed when a coupler's flux pulse changes its EJ so
much that its instantaneous states are spread over many fixed idle-basis levels;
a uniform cap would then leave different coupler levels to different
computational sectors). Each two-mode term is applied jointly before projection,
P (n_i n_j) P, never (P n_i P)(P n_j P). Cap N=None is the full tensor product.
The cap is a numerical truncation whose adequacy must be shown by a cap ladder.

Local energy eigenstates at ng=0 have definite reflection parity equal to their
index parity, so n changes index parity and cos(phi) preserves it. Roundoff
entries that violate this selection rule are removed and their maximum recorded.

Numerical model only: no field extraction, calibration, loss or hardware claim.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh

CELL_SCHEMA = 'emulateqpu.chain.cell.v1'
LINE_KEYS = ('tau_ns', 'gain', 'settling_time_constants', 'rise_fraction')
DENSE_ENCODING_LIMIT = 3000
PARITY_ROUNDOFF_LIMIT = 1e-10


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def array_digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode()); h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- local physics
# Exact copy of emulateqpu_twoq.core._kernel._local (itself an exact slice of
# digital_twin/tunable_coupler.py:_local). Do not edit: reproduction depends on it.
def _local(EC, EJ, L, ncut):
    n = np.arange(-ncut, ncut+1, dtype=float)
    energies, V = eigh_tridiagonal(4*EC*n*n, np.full(2*ncut, -EJ/2), select='i', select_range=(0, L-1))
    for k in range(1, L):
        if np.dot(V[:, k-1]*n, V[:, k]) < 0: V[:, k] *= -1
    cosine = np.diag(np.full(2*ncut, .5), 1)+np.diag(np.full(2*ncut, .5), -1)
    return energies-energies[0], V.T@(n[:, None]*V), V.T@cosine@V


# Exact copy of emulateqpu_twoq.core._kernel.squid_EJ_GHz.
def squid_EJ_GHz(EJmax_GHz, flux):
    """Symmetric, zero-loop-inductance SQUID, Phi/Phi0 on |Phi|<.5 branch."""
    x = np.asarray(flux, float)
    if not np.isfinite(x).all() or np.any(np.abs(x) >= .5): raise ValueError('Flux is outside |Phi/Phi0|<.5')
    return float(EJmax_GHz)*np.cos(np.pi*x)


def _real(value, name, lo=None, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
        raise ValueError(name+' must be a finite real scalar, excluding bool')
    v = float(value)
    if (lo is not None and v < lo) or (hi is not None and v > hi): raise ValueError(name+' is out of range')
    return v


def validate_cell(cell):
    """Explicit chain cell. Every physical number is supplied; nothing is retargeted."""
    if not isinstance(cell, dict): raise ValueError('Explicit cell object required')
    c = deepcopy(cell)
    allowed = {'schema', 'name', 'mode_labels', 'mode_roles', 'EC_matrix_GHz', 'EJ_idle_GHz', 'idle_flux_Phi0',
               'ncut', 'lines', 'parameter_origin', 'provenance', 'assumptions'}
    if set(c)-allowed: raise ValueError('Unknown cell fields: '+str(sorted(set(c)-allowed)))
    if c.get('schema') != CELL_SCHEMA: raise ValueError('Unsupported cell schema')
    roles = c.get('mode_roles')
    if not isinstance(roles, list) or not roles or any(r not in ('data', 'coupler') for r in roles):
        raise ValueError('mode_roles must be a nonempty list of data/coupler roles')
    m = len(roles)
    labels = c.get('mode_labels')
    if not isinstance(labels, list) or len(labels) != m or len(set(labels)) != m or not all(isinstance(x, str) and x for x in labels):
        raise ValueError('One unique nonempty label per mode required')
    if 'data' not in roles: raise ValueError('At least one data mode required')
    ec = np.asarray(c.get('EC_matrix_GHz'), dtype=float)
    if ec.shape != (m, m) or not np.isfinite(ec).all() or not np.allclose(ec, ec.T, rtol=0, atol=1e-13) or np.linalg.eigvalsh(ec)[0] <= 0:
        raise ValueError('EC_matrix_GHz must be a finite symmetric positive-definite m x m matrix')
    c['EC_matrix_GHz'] = ec.tolist()
    ej = np.asarray(c.get('EJ_idle_GHz'), dtype=float)
    if ej.shape != (m,) or not np.isfinite(ej).all() or np.any(ej <= 0) or np.any(ej > 500):
        raise ValueError('One EJ/h in (0,500] GHz per mode required')
    c['EJ_idle_GHz'] = ej.tolist()
    flux = c.get('idle_flux_Phi0', [0.]*m)
    if not isinstance(flux, list) or len(flux) != m: raise ValueError('One idle flux per mode required')
    flux = [_real(x, 'idle flux', 0., .45) for x in flux]
    if any(f != 0 for f, r in zip(flux, roles) if r == 'data'): raise ValueError('Fixed-frequency data modes have no flux')
    if any(f >= .45 for f in flux): raise ValueError('Idle flux must stay below .45 Phi0')
    c['idle_flux_Phi0'] = flux
    if type(c.get('ncut')) is not int or not 16 <= c['ncut'] <= 48: raise ValueError('Charge cutoff must be an integer 16..48')
    lines = c.get('lines', {})
    if not isinstance(lines, dict): raise ValueError('lines must map coupler labels to flux-line recipes')
    couplers = [labels[i] for i in range(m) if roles[i] == 'coupler']
    if set(lines) != set(couplers): raise ValueError('Exactly one declared flux-line recipe per coupler required')
    for name in couplers:
        line = lines[name]
        if not isinstance(line, dict) or set(line) != set(LINE_KEYS): raise ValueError('Complete line recipe required for '+name)
        lines[name] = {'tau_ns': _real(line['tau_ns'], 'line tau', 0., 5.), 'gain': _real(line['gain'], 'line gain', .01, 2.),
            'settling_time_constants': _real(line['settling_time_constants'], 'settling', 3., 10.),
            'rise_fraction': _real(line['rise_fraction'], 'rise fraction', .05, .5)}
    c['lines'] = lines
    origin = c.get('parameter_origin', 'assumed_circuit')
    if origin not in ('assumed_circuit', 'extracted_full_cell', 'assembled_from_cells', 'supplied_fit', 'calibrated'):
        raise ValueError('Unsupported parameter_origin')
    c['parameter_origin'] = origin
    for name in ('provenance', 'assumptions'):
        if name in c and not isinstance(c[name], dict): raise ValueError(name+' must be an object')
    canonical(c)
    return c


# ---------------------------------------------------------------- product basis
@dataclass(frozen=True)
class Basis:
    dims: tuple
    cap: int | None
    states: np.ndarray        # (D, m) local indices, lexicographic (= full-tensor order)
    codes: np.ndarray         # full-tensor flat index, strictly increasing
    strides: np.ndarray
    parity: np.ndarray        # sum(local indices) % 2
    weights: tuple = None     # cap weights (None = all ones); 0 marks a cap-exempt mode

    @property
    def dimension(self): return len(self.codes)

    @property
    def weighted_excitations(self):
        w = np.ones(len(self.dims), dtype=np.int64) if self.weights is None else np.asarray(self.weights, dtype=np.int64)
        return self.states.astype(np.int64)@w

    def lookup(self, codes):
        codes = np.asarray(codes, dtype=np.int64)
        pos = np.searchsorted(self.codes, codes)
        ok = pos < len(self.codes)
        ok[ok] = self.codes[pos[ok]] == codes[ok]
        return np.where(ok, pos, -1)


def basis_dimension(dims, cap=None, weights=None):
    """Exact dimension using scalar counts, before allocating any basis rows."""
    dims = tuple(dims)
    if not dims or any(type(d) is not int or d < 1 for d in dims):
        raise ValueError('Local dimensions must be positive integers')
    weights = (1,)*len(dims) if weights is None else tuple(weights)
    if len(weights) != len(dims) or any(type(w) is not int or w not in (0, 1) for w in weights):
        raise ValueError('Cap weights must be 0 or 1, one per mode')
    if cap is None:
        return math.prod(dims)
    if type(cap) is not int or cap < 0:
        raise ValueError('Excitation cap must be a nonnegative integer or None')
    counts = {0: 1}
    for d, w in zip(dims, weights):
        new = {}
        for old, count in counts.items():
            for level in range(min(d, (cap-old)//w+1) if w else d):
                total = old+w*level
                new[total] = new.get(total, 0)+count
        counts = new
    return sum(counts.values())


def make_basis(dims, cap=None, weights=None):
    dims = tuple(int(d) for d in dims)
    if any(d < 1 for d in dims): raise ValueError('Local dimensions must be positive')
    if weights is None or all(int(w) == 1 for w in weights): weights = None
    else:
        weights = tuple(int(w) for w in weights)
        if len(weights) != len(dims) or any(w not in (0, 1) for w in weights): raise ValueError('Cap weights must be 0 or 1, one per mode')
    w = np.ones(len(dims), dtype=np.int64) if weights is None else np.asarray(weights, dtype=np.int64)
    dimension = basis_dimension(dims, cap, tuple(int(x) for x in w))
    if dimension > 2_000_000:
        raise ValueError('Retained basis exceeds 2,000,000 states before allocation')
    if math.prod(dims) > np.iinfo(np.int64).max:
        raise ValueError('Full-tensor basis codes exceed the supported int64 representation')
    if cap is not None:
        if type(cap) is not int or cap < 0: raise ValueError('Excitation cap must be a nonnegative integer or None')
        if cap >= sum(wi*(d-1) for wi, d in zip(w, dims)): cap = None   # no truncation: identical to the full tensor
    if cap is None: weights = None
    strides = np.array([math.prod(dims[i+1:]) for i in range(len(dims))], dtype=np.int64)
    rows = np.zeros((1, 0), dtype=np.int16)
    for k, d in enumerate(dims):
        sums = rows.astype(np.int64)@w[:k]
        blocks = []
        for v in range(d):
            keep = rows[sums+w[k]*v <= cap] if cap is not None else rows
            blocks.append(np.hstack([keep, np.full((len(keep), 1), v, dtype=np.int16)]))
        rows = np.vstack(blocks)
        if len(rows) > 2_000_000: raise ValueError('Retained basis exceeds 2,000,000 states')
    codes = rows.astype(np.int64)@strides
    order = np.argsort(codes)
    rows, codes = rows[order], codes[order]
    rows.setflags(write=False); codes.setflags(write=False)
    return Basis(dims, cap, rows, codes, strides, (rows.sum(axis=1) % 2).astype(np.int8), weights)


def _parity_clean(A, odd):
    """Zero roundoff entries that violate the ng=0 reflection selection rule."""
    i, j = np.indices(A.shape)
    forbidden = ((i-j) % 2 == 0) if odd else ((i-j) % 2 == 1)
    dropped = float(np.max(np.abs(A[forbidden]))) if forbidden.any() else 0.
    B = A.copy(); B[forbidden] = 0.
    return B, dropped


def one_mode_operator(basis, i, A, coefficient=1.):
    """P (coefficient * A_i) P on the capped basis, as CSR (row=target, col=source)."""
    S, codes, D = basis.states, basis.codes, basis.dimension
    si = S[:, i].astype(np.int64)
    rows, cols, vals = [], [], []
    for ap in range(A.shape[0]):
        v = A[ap, si]
        src = np.flatnonzero(v != 0)
        if not len(src): continue
        pos = basis.lookup(codes[src]+(ap-si[src])*basis.strides[i])
        ok = pos >= 0
        rows.append(pos[ok]); cols.append(src[ok]); vals.append(coefficient*v[src[ok]])
    if not rows: return sp.csr_matrix((D, D))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D))


def two_mode_operator(basis, i, A, j, B, coefficient):
    """P (coefficient * A_i B_j) P, the jointly projected product; no intermediate projector."""
    S, codes, D = basis.states, basis.codes, basis.dimension
    si = S[:, i].astype(np.int64); sj = S[:, j].astype(np.int64)
    rows, cols, vals = [], [], []
    for ap in range(A.shape[0]):
        a = A[ap, si]
        if not np.any(a): continue
        for bp in range(B.shape[0]):
            v = a*B[bp, sj]
            src = np.flatnonzero(v != 0)
            if not len(src): continue
            pos = basis.lookup(codes[src]+(ap-si[src])*basis.strides[i]+(bp-sj[src])*basis.strides[j])
            ok = pos >= 0
            rows.append(pos[ok]); cols.append(src[ok]); vals.append(coefficient*v[src[ok]])
    if not rows: return sp.csr_matrix((D, D))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D))


# ---------------------------------------------------------------- bound model
@dataclass
class ChainModel:
    cell: dict
    basis: Basis
    local: list                 # per mode: (energies, n, cos) retained local matrices (parity-cleaned)
    H: sp.csr_matrix            # idle H/h, real
    cos_ops: dict               # coupler mode index -> P cos P (real CSR)
    charge_ops: dict            # mode index -> P n P (real CSR)
    data_modes: tuple
    coupler_modes: tuple
    comp_rows: np.ndarray       # basis rows of bare computational tuples, bit order q_first = most significant
    encoding: np.ndarray        # (D, 2^n) dressed computational isometry
    encoded_energies: np.ndarray
    analysis: dict = field(default_factory=dict)

    @property
    def EC(self): return np.asarray(self.cell['EC_matrix_GHz'])

    @property
    def EJ(self): return np.asarray(self.cell['EJ_idle_GHz'])

    def EJmax(self, mode):
        return float(self.EJ[mode]/np.cos(np.pi*self.cell['idle_flux_Phi0'][mode]))

    def identity(self):
        return dict(cell_sha256=digest(self.cell), dims=list(self.basis.dims), cap=self.basis.cap,
            cap_weights=None if self.basis.weights is None else list(self.basis.weights),
            operator_sha256=array_digest({'H_data': self.H.data, 'H_indices': self.H.indices, 'H_indptr': self.H.indptr,
                                          'encoding': self.encoding, 'energies': self.encoded_energies}))


def _computational_rows(basis, data_modes, labels=None):
    m = len(basis.dims); out = []
    bits_list = np.ndindex((2,)*len(data_modes)) if labels is None else [tuple(map(int, s)) for s in labels]
    for bits in bits_list:
        t = np.zeros(m, dtype=np.int64)
        for q, b in zip(data_modes, bits): t[q] = b
        r = basis.lookup([int(t@basis.strides)])[0]
        if r < 0: raise ValueError('Excitation cap excludes a computational state')
        out.append(r)
    return np.array(out)


def _phase_fix(V, comp_rows):
    V = V.astype(complex)
    for k, b in enumerate(comp_rows): V[:, k] *= np.exp(-1j*np.angle(V[b, k]))
    return V


def _davidson_state(H, diag, b, tol=1e-11, max_basis=48, keep=8, max_iter=4000):
    """Interior eigenvector adiabatically connected to bare row b (overlap-tracking Davidson).

    Search space starts at e_b; corrections are diagonal-preconditioned residuals
    (theta - H_kk)^-1 r. The Ritz pair kept at each step is the one with the largest
    weight on row b, not the one nearest a shift, so a near-degenerate non-computational
    neighbour cannot capture the label. H preserves index parity, so the space stays in
    b's parity block automatically.
    """
    D = H.shape[0]
    V = np.zeros((D, max_basis)); V[b, 0] = 1.; HV = np.zeros((D, max_basis)); HV[:, 0] = H@V[:, 0]; m = 1
    theta = H[b, b]; x = V[:, 0]
    for it in range(max_iter):
        S = V[:, :m].T@HV[:, :m]; S = (S+S.T)/2
        w, Y = np.linalg.eigh(S)
        k = int(np.argmax((V[b, :m]@Y)**2))
        theta = w[k]; y = Y[:, k]; x = V[:, :m]@y; Hx = HV[:, :m]@y
        r = Hx-theta*x; nr = np.linalg.norm(r)
        if nr < tol: return theta, x/np.linalg.norm(x), dict(iterations=it, residual=float(nr))
        if m == max_basis:
            order = np.argsort(-(V[b, :m]@Y)**2)[:keep]
            Q = V[:, :m]@Y[:, order]; HQ = HV[:, :m]@Y[:, order]
            V[:, :keep] = Q; HV[:, :keep] = HQ; V[:, keep:] = 0; HV[:, keep:] = 0; m = keep
        den = theta-diag
        den = np.where(np.abs(den) < 1e-3, np.copysign(1e-3, den+1e-300), den)
        t = r/den
        for _ in range(2): t -= V[:, :m]@(V[:, :m].T@t)
        nt = np.linalg.norm(t)
        if nt < 1e-14: t = r-V[:, :m]@(V[:, :m].T@r); nt = np.linalg.norm(t)
        V[:, m] = t/nt; HV[:, m] = H@V[:, m]; m += 1
    raise RuntimeError('Davidson encoding did not converge for row %d (residual %.3g)' % (b, nr))


def dressed_encoding(H, basis, comp_rows, method='auto', candidates=8):
    """Dressed idle eigenvectors with maximal overlap on bare computational tuples.

    Dense: one global assignment over every eigenvector, as in the two-qubit kernel.
    Sparse: shift-invert candidates near each bare energy inside its exact parity
    block, followed by one global assignment over the union of candidates.
    """
    D = H.shape[0]; n = len(comp_rows)
    if method == 'auto': method = 'dense' if D <= DENSE_ENCODING_LIMIT else 'sparse'
    if method == 'davidson':
        diag = H.diagonal().astype(float); vecs, vals, its = [], [], []
        for b in comp_rows:
            lam, x, info = _davidson_state(H, diag, int(b))
            vecs.append(x); vals.append(lam); its.append(info['iterations'])
        V = np.array(vecs).T; vals = np.array(vals)
        P = np.abs(V[comp_rows, :])**2
        if not np.array_equal(np.argmax(P, axis=0), np.arange(n)):
            raise ValueError('Davidson encoding: a dressed state is not dominated by its own bare computational row')
        gram = np.abs(V.T@V-np.eye(n)).max()
        if gram > 1e-8: raise ValueError('Davidson encoding: two rows converged to the same eigenvector (gram %.3g)' % gram)
        enc = _phase_fix(V, comp_rows)
        return enc, vals, dict(method='overlap-tracking Davidson per computational row (diagonal preconditioner)', candidates=int(n),
                               iterations=[int(x) for x in its])
    if method == 'dense':
        energies, V = eigh(H.toarray())
        r, c = linear_sum_assignment(-np.abs(V[comp_rows, :])**2)
        cols = c[np.argsort(r)]
        enc = _phase_fix(V[:, cols], comp_rows)
        return enc, energies[cols], dict(method='dense eigh, global assignment', candidates=int(D))
    vecs, vals = [], []
    for b in comp_rows:
        p = basis.parity[b]
        rows = np.flatnonzero(basis.parity == p)
        Hp = H[rows][:, rows]
        k = min(candidates, len(rows)-2)
        w, X = eigsh(Hp.tocsc(), k=k, sigma=float(H[b, b]), which='LM', tol=1e-13)
        for lam, x in zip(w, X.T):
            full = np.zeros(D); full[rows] = x
            if any(abs(lam-l) < 1e-11 and abs(abs(np.dot(full, v)) - 1) < 1e-8 for l, v in zip(vals, vecs)): continue
            vecs.append(full); vals.append(lam)
    V = np.array(vecs).T; vals = np.array(vals)
    r, c = linear_sum_assignment(-np.abs(V[comp_rows, :])**2)
    cols = c[np.argsort(r)]
    enc = _phase_fix(V[:, cols], comp_rows)
    return enc, vals[cols], dict(method='parity-block shift-invert eigsh, union assignment', candidates=int(len(vals)))


def build_model(cell, dims, cap=None, encoding_method='auto', cap_exempt=(), *, encoding_labels=None, encoding_kind='dressed'):
    c = validate_cell(cell)
    roles = c['mode_roles']; m = len(roles)
    dims = tuple(int(d) for d in dims)
    if len(dims) != m: raise ValueError('One retained dimension per mode required')
    data = tuple(i for i in range(m) if roles[i] == 'data'); couplers = tuple(i for i in range(m) if roles[i] == 'coupler')
    if encoding_kind not in ('dressed', 'bare'):
        raise ValueError('encoding_kind must be dressed or bare')
    if encoding_labels is not None:
        if (not isinstance(encoding_labels, (list, tuple)) or not encoding_labels
            or any(not isinstance(s, str) or len(s) != len(data) or set(s)-{'0', '1'} for s in encoding_labels)
            or len(set(encoding_labels)) != len(encoding_labels)):
            raise ValueError('encoding_labels must be unique data-bit strings in explicit column order')
        encoding_labels = list(encoding_labels)
    if any(dims[i] < 3 for i in data): raise ValueError('Data modes need at least three retained levels to expose leakage')
    cap_exempt = tuple(cap_exempt)
    if any(l not in c['mode_labels'] or roles[c['mode_labels'].index(l)] != 'coupler' for l in cap_exempt):
        raise ValueError('Only coupler modes can be exempt from the excitation cap')
    weights = [0 if c['mode_labels'][i] in cap_exempt else 1 for i in range(m)]
    basis = make_basis(dims, cap, weights)
    ec = np.asarray(c['EC_matrix_GHz']); ej = np.asarray(c['EJ_idle_GHz'])
    local, dropped = [], 0.
    for i in range(m):
        E, n, cs = _local(ec[i, i], ej[i], dims[i], c['ncut'])
        n2, d1 = _parity_clean(n, odd=True); cs2, d2 = _parity_clean(cs, odd=False)
        dropped = max(dropped, d1, d2)
        local.append((E, n2, cs2))
    if dropped > PARITY_ROUNDOFF_LIMIT: raise ValueError('Local operators violate the ng=0 parity selection rule beyond roundoff')
    diag = np.zeros(basis.dimension)
    for i in range(m): diag += local[i][0][basis.states[:, i]]
    H = sp.diags(diag).tocsr()
    for i in range(m):
        for j in range(i+1, m):
            if ec[i, j] != 0: H = H+two_mode_operator(basis, i, local[i][1], j, local[j][1], 8*ec[i, j])
    H = H.tocsr(); H.sum_duplicates(); H.sort_indices()
    asym = abs(H-H.T).max() if H.nnz else 0.
    if asym > 1e-13: raise ValueError('Assembled Hamiltonian is not symmetric')
    cos_ops = {i: one_mode_operator(basis, i, local[i][2]) for i in couplers}
    charge_ops = {i: one_mode_operator(basis, i, local[i][1]) for i in range(m)}
    comp = _computational_rows(basis, data, encoding_labels)
    if encoding_kind == 'bare':
        enc = np.zeros((basis.dimension, len(comp)), complex)
        enc[comp, np.arange(len(comp))] = 1.
        energies = np.asarray(H.diagonal()[comp])
        enc_info = dict(method='explicit bare product columns; not an idle eigenspace', candidates=len(comp))
    else:
        enc, energies, enc_info = dressed_encoding(H, basis, comp, encoding_method)
    resid = np.abs(H@enc-enc*energies).max()
    if encoding_kind == 'dressed' and resid > 1e-8: raise ValueError('Dressed encoding is not an idle eigenspace (residual %.3g)' % resid)
    gram = np.abs(enc.conj().T@enc-np.eye(len(comp))).max()
    if gram > 1e-9: raise ValueError('Dressed encoding is not an isometry')
    overlap = np.abs(enc[comp, np.arange(len(comp))])**2
    spectra = [_local(ec[i, i], ej[i], 3, c['ncut']) for i in range(m)]
    n01 = np.array([x[1][0, 1] for x in spectra])
    exchange = 8*ec*np.outer(n01, n01)*1000; np.fill_diagonal(exchange, 0)
    nd = len(data)
    def bitstring(k): return encoding_labels[k] if encoding_labels is not None else ''.join(str(b) for b in np.unravel_index(k, (2,)*nd))
    analysis = dict(mode_labels=c['mode_labels'], mode_roles=roles, dims=list(dims), excitation_cap=basis.cap,
        excitation_cap_weights=None if basis.weights is None else list(basis.weights),
        cap_exempt=[l for l, w in zip(c['mode_labels'], weights) if w == 0 and basis.weights is not None],
        dimension=int(basis.dimension), full_tensor_dimension=math.prod(dims), nnz=int(H.nnz),
        local_f01_GHz=[float(x[0][1]) for x in spectra],
        local_anharmonicity_MHz=[float((x[0][2]-2*x[0][1])*1000) for x in spectra],
        local_n01=n01.tolist(), idle_exchange_MHz=exchange.tolist(),
        spectral_upper_bound_GHz=float(diag.max()),
        parity_roundoff_dropped_max=dropped, encoding=enc_info, encoding_residual_GHz=float(resid), encoding_gram_error=float(gram),
        dressed_bare_overlap={bitstring(k): float(v) for k, v in enumerate(overlap)},
        dressed_energies_GHz={bitstring(k): float(v) for k, v in enumerate(energies)},
        computational_bit_order='q-labels of data modes in chain order; first data mode is the most significant bit',
        units={'H': 'H/h GHz', 'time': 'ns', 'flux': 'Phi/Phi0'},
        parameter_origin=c['parameter_origin'], physical_validation=False)
    if encoding_labels is not None or encoding_kind != 'dressed':
        analysis.update(encoding_labels=encoding_labels or [bitstring(k) for k in range(len(comp))],
                        encoding_kind=encoding_kind, complete_computational_encoding=(len(comp) == 2**nd))
    return ChainModel(c, basis, local, H, cos_ops, charge_ops, data, couplers, comp, enc, energies, analysis)


def static_zz_MHz(model):
    """Pairwise static ZZ between data qubits from dressed energies, others in 0."""
    if model.analysis.get('encoding_kind', 'dressed') != 'dressed' or model.encoding.shape[1] != 2**len(model.data_modes):
        raise ValueError('Static ZZ requires the complete dressed computational encoding')
    E = model.encoded_energies; nd = len(model.data_modes); out = {}
    selected_labels = model.analysis.get('encoding_labels')
    labels = [model.cell['mode_labels'][q] for q in model.data_modes]
    for a in range(nd):
        for b in range(a+1, nd):
            def idx(bits):
                v = [0]*nd
                for q, x in bits.items(): v[q] = x
                return selected_labels.index(''.join(map(str, v))) if selected_labels is not None else int(np.ravel_multi_index(v, (2,)*nd))
            zz = E[idx({a: 1, b: 1})]-E[idx({a: 1})]-E[idx({b: 1})]+E[idx({})]
            out[labels[a]+'-'+labels[b]] = float(zz*1000)
    return out
