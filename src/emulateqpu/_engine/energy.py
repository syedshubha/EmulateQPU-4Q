"""Energy observables of the idle chain Hamiltonian (0.1.0.dev4).

H/h = sum_i H_i + sum_{i<j} V_ij exactly as build_model assembles it:
- H_i is the retained local eigenenergy diagonal of mode i (zero at its local ground state);
- V_ij is the jointly projected P (8 EC_ij n_i n_j) P.
The decomposition is checked against model.H to roundoff.

Region energies for QET-style accounting use one of two declared conventions (both reported; no single convention is
chosen silently):
- 'local': sum_{i in R} H_i;
- 'half_bond': sum_{i in R} H_i + sum_{i<j in R} V_ij + 1/2 sum_{i in R, j not in R} V_ij.
Every value is reported relative to the same term in the dressed idle ground state (encoding column 0), in h*MHz.
These are expectation values of the idle Hamiltonian; they are meaningful when every flux pulse is back at idle.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .model import two_mode_operator

CONVENTIONS = ('local', 'half_bond')


class EnergyTerms:
    def __init__(self, model):
        self.model = model
        labels = model.cell['mode_labels']; S = model.basis.states; ec = model.EC; m = len(labels)
        self.local = {labels[i]: model.local[i][0][S[:, i]].astype(float) for i in range(m)}
        self.bonds = {}
        for i in range(m):
            for j in range(i+1, m):
                if ec[i, j] != 0:
                    self.bonds[(labels[i], labels[j])] = two_mode_operator(model.basis, i, model.local[i][1], j, model.local[j][1], 8*ec[i, j]).tocsr()
        total = sp.diags(sum(self.local.values())).tocsr()
        for B in self.bonds.values(): total = total+B
        self.decomposition_error_GHz = float(abs(total-model.H).max())
        if self.decomposition_error_GHz > 1e-10: raise ValueError('Energy terms do not reproduce the idle Hamiltonian')
        selected = model.analysis.get('encoding_labels')
        ground_label = '0'*len(model.data_modes)
        if model.analysis.get('encoding_kind', 'dressed') == 'dressed' and (selected is None or ground_label in selected):
            column = 0 if selected is None else selected.index(ground_label)
            self.reference = dict(kind='dressed_idle_ground', label=ground_label, column=column)
        else:
            column = 0
            self.reference = dict(kind='first_selected_encoding_column', label=None if selected is None else selected[0], column=column,
                                  is_dressed_idle_ground=False)
        g = model.encoding[:, column]
        self.ground = {k: float(np.real(np.vdot(g, v*g))) for k, v in self.local.items()}
        self.ground.update({k: float(np.real(np.vdot(g, B@g))) for k, B in self.bonds.items()})

    def term_values(self, branch):
        """Unnormalized Tr(term rho) for every term (GHz * probability)."""
        V, W = branch.V, branch.W
        out = {}
        d = branch.diagonal()
        for k, v in self.local.items(): out[k] = float(np.dot(v, d))
        for k, B in self.bonds.items(): out[k] = float(np.real(np.trace(W@(V.conj().T@(B@V)))))
        return out

    def report(self, branch, regions=None):
        """Conditional (normalized) term energies relative to the ground state, in MHz, plus region sums."""
        p = branch.probability()
        if p <= 0: return dict(probability=p)
        raw = self.term_values(branch)
        rel = {k: (v/p-self.ground[k])*1000 for k, v in raw.items()}
        total = sum(rel.values())
        out = dict(probability=p, total_MHz=total,
                   local_MHz={k: rel[k] for k in self.local},
                   bond_MHz={'~'.join(k): rel[k] for k in self.bonds})
        if regions:
            out['regions_MHz'] = {}
            for name, modes in regions.items():
                modes = set(modes)
                loc = sum(rel[k] for k in self.local if k in modes)
                inside = sum(rel[k] for k in self.bonds if k[0] in modes and k[1] in modes)
                cross = sum(rel[k] for k in self.bonds if (k[0] in modes) != (k[1] in modes))
                out['regions_MHz'][name] = dict(local=loc, half_bond=loc+inside+.5*cross, internal_bonds=inside, boundary_bonds=cross)
        return out
