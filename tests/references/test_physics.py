"""Independent numerical references for small finite superconducting models."""

import unittest

import numpy as np
import qutip
import scqubits

from emulateqpu.chain import linear_chain_cell
from emulateqpu._engine.evolve import propagate_states
from emulateqpu._engine.model import _local, build_model


class PhysicsReferenceTests(unittest.TestCase):
    def test_cosine_transmon_energies_match_scqubits(self):
        for charging, junction in ((0.24, 14.0), (0.30, 9.0), (0.20, 20.0)):
            with self.subTest(EC=charging, EJ=junction):
                levels = 5
                cutoff = 24
                reference = scqubits.Transmon(EJ=junction, EC=charging, ng=0, ncut=cutoff)
                energies = reference.eigenvals(evals_count=levels)
                actual, _, _ = _local(charging, junction, levels, cutoff)
                np.testing.assert_allclose(actual, energies - energies[0], atol=2e-10, rtol=0)

    def test_carrier_resolved_states_match_qutip(self):
        model = build_model(linear_chain_cell(2, coupling_GHz=0.001), (3, 3, 3))
        initial = (model.encoding[:, 0] + model.encoding[:, 2]) / np.sqrt(2)
        amplitude, carrier, phase = 0.04, 4.72, 0.30
        duration, offset = 0.42, 0.17
        drive = dict(mode="q0", amplitude_GHz=amplitude, frequency_GHz=carrier,
                     phase_rad=phase, start_ns=0.0, duration_ns=duration, envelope="square")
        schedule = dict(duration_ns=duration, drives=[drive], flux=[])
        actual = propagate_states(model, schedule, initial[:, None], samples=9, keep_states=True,
                                  carrier_time_offset_ns=offset, rtol=1e-11, atol=1e-13)

        # The application supplies H/h and charge in its retained basis. QuTiP
        # integrates the independent lab-frame coefficient with angular units.
        def microwave(t):
            return amplitude * np.cos(2 * np.pi * carrier * (t + offset) + phase)

        h0 = qutip.Qobj(2 * np.pi * model.H.toarray())
        charge = qutip.Qobj(2 * np.pi * model.charge_ops[0].toarray())
        hamiltonian = qutip.QobjEvo([h0, [charge, microwave]])
        result = qutip.sesolve(hamiltonian, qutip.Qobj(initial), actual["times_ns"], options={
            "method": "vern9",
            "rtol": 1e-11,
            "atol": 1e-13,
            "max_step": 1 / (20 * carrier),
            "normalize_output": False,
        })
        expected = np.stack([state.full()[:, 0] for state in result.states])
        np.testing.assert_allclose(actual["states"][:, :, 0], expected, atol=3e-9, rtol=0)


if __name__ == "__main__":
    unittest.main()
