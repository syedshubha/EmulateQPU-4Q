"""Gate metrics on the dressed computational block of n data qubits.

M[k_out, k_in] = <dressed k_out | psi_k_in(T)>. Bit order: first data mode is
the most significant bit. Fidelity is the unconditional Haar average
(Tr(M^dag M) + |Tr(U^dag Z M)|^2) / (d(d+1)), d = 2^n, with Z the declared
software output frame diag exp(-i sum_q beta_q b_q). Leaked probability is
retained as loss; nothing is postselected. For d = 4 this is the two-qubit
kernel's (4 s + |Tr|^2)/20 exactly.
"""
from __future__ import annotations

import numpy as np


def _bits(nd):
    return np.array(list(np.ndindex((2,)*nd)))


def single_index(nd, q):
    v = [0]*nd; v[q] = 1
    return int(np.ravel_multi_index(v, (2,)*nd))


def cz_target(nd, a, b):
    B = _bits(nd)
    return np.diag(np.where((B[:, a] == 1) & (B[:, b] == 1), -1., 1.)).astype(complex)


def local_frame_from_diagonal(M):
    """beta_q = arg(M[e_q,e_q] M[0,0]^*): the inherited default frame rule."""
    nd = int(round(np.log2(M.shape[0]))); d = np.diag(M)
    return np.array([np.angle(d[single_index(nd, q)]*d[0].conj()) for q in range(nd)])


def gate_metrics(M, U, output_phases=None):
    d = M.shape[0]; nd = int(round(np.log2(d)))
    beta = local_frame_from_diagonal(M) if output_phases is None else np.asarray(output_phases, float)
    Z = np.exp(-1j*(_bits(nd)@beta))
    Mc = Z[:, None]*M
    gram = M.conj().T@M
    survival = float(np.trace(gram).real/d)
    F = float((d*survival+abs(np.trace(U.conj().T@Mc))**2)/(d*(d+1)))
    leak = 1-np.sum(np.abs(M)**2, axis=0)
    return dict(average_gate_fidelity_unconditional=F, mean_leakage=float(1-survival),
        worst_input_leakage=float(leak.max()), worst_state_leakage=float(1-np.linalg.eigvalsh(gram).min()),
        leakage_per_input=leak.tolist(), output_phases_rad=beta.tolist(),
        computational_offdiagonal_probability=float(np.sum(np.abs(M-np.diag(np.diag(M)))**2)/d))


def conditional_phases(M, a, b):
    """CZ conditional phase on data pair (a,b) for every spectator configuration."""
    nd = int(round(np.log2(M.shape[0]))); others = [q for q in range(nd) if q not in (a, b)]
    d = np.diag(M); out = {}
    for spec in np.ndindex((2,)*len(others)):
        def idx(xa, xb):
            v = [0]*nd; v[a] = xa; v[b] = xb
            for q, x in zip(others, spec): v[q] = x
            return int(np.ravel_multi_index(v, (2,)*nd))
        theta = float(np.angle(d[idx(1, 1)]*d[idx(0, 0)]*d[idx(0, 1)].conj()*d[idx(1, 0)].conj()))
        out[''.join(map(str, spec))] = dict(theta_rad=theta, error_from_pi_rad=float(np.angle(np.exp(1j*(theta-np.pi)))))
    return out


def spectator_conditioned_pair(M, a, b, output_phases):
    """Pair-block fidelity to CZ for each fixed computational spectator configuration.

    The pair block keeps only rows/columns with the spectator bits fixed, so
    spectator flips and leakage both count as loss of that block.
    """
    nd = int(round(np.log2(M.shape[0]))); others = [q for q in range(nd) if q not in (a, b)]
    beta = np.asarray(output_phases, float)
    B = _bits(nd); Z = np.exp(-1j*(B@beta)); Mc = Z[:, None]*M
    U = np.diag([1, 1, 1, -1]).astype(complex); out = {}
    for spec in np.ndindex((2,)*len(others)):
        sel = [k for k in range(2**nd) if all(B[k, q] == x for q, x in zip(others, spec))]
        sel = sorted(sel, key=lambda k: (B[k, a], B[k, b]))
        P = Mc[np.ix_(sel, sel)]
        s = float(np.trace(P.conj().T@P).real)
        out[''.join(map(str, spec))] = dict(fidelity=float((s+abs(np.trace(U.conj().T@P))**2)/20), survival=s/4)
    return out
