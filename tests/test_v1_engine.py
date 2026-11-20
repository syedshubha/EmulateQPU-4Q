"""Bounded integration contracts for the supported engine changes."""
import copy
import unittest
from unittest import mock

import numpy as np

from emulateqpu._engine.fixtures import demo_cell
from emulateqpu._engine.model import build_model, basis_dimension
from emulateqpu._engine.model import static_zz_MHz
from emulateqpu._engine.evolve import propagate_states
from emulateqpu._engine import measurement as ms
from emulateqpu._engine.program import compile_program, run_program, measurement_interval_ns
from emulateqpu.chain import linear_chain_cell, mode_order


class EngineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model(demo_cell(), (3, 3, 3))

    def measurement(self, **changes):
        op = dict(op='measure', key='m', qubits=['q0'],
                  instrument=dict(model='dispersive_qnd_heterodyne', duration_ns=300., drive_duration_ns=200., response_order=16))
        op.update(changes)
        return op

    def compile_measure(self, op):
        return compile_program(self.model, dict(feedback_latency_ns=200., operations=[op,
            dict(op='wait', duration_ns=0., condition=dict(key='m', equals='0'))]))

    def test_inconsistent_measurement_rejected_before_response_allocation(self):
        with mock.patch.object(ms, 'qnd_instrument', side_effect=AssertionError('must not allocate response')):
            with self.assertRaisesRegex(ValueError, 'response interval'):
                self.compile_measure(self.measurement(duration_ns=100.))

    def test_matching_measurement_and_feedback_endpoint(self):
        _, stages, total = self.compile_measure(self.measurement(duration_ns=300.))
        self.assertEqual(stages[0]['end_ns'], 300.)
        self.assertEqual(stages[-1]['start_ns'], 500.)
        self.assertEqual(total, 500.)

    def test_omitted_duration_uses_instrument(self):
        _, stages, _ = self.compile_measure(self.measurement())
        self.assertEqual(stages[0]['duration_ns'], 300.)
        self.assertEqual(stages[-1]['start_ns'], 500.)

    def test_concurrent_duration_mismatch(self):
        op = self.measurement(concurrent=dict(duration_ns=100., drives=[], flux=[]))
        with self.assertRaisesRegex(ValueError, 'Concurrent'):
            self.compile_measure(op)

    def test_readout_drive_must_fit_response(self):
        op = self.measurement()
        op['instrument']['drive_duration_ns'] = 301.
        with self.assertRaises(ValueError):
            measurement_interval_ns(op)

    def test_zero_duration_ideal_measurement(self):
        p = dict(operations=[dict(op='measure', key='m', qubits=['q0'],
                                 instrument=dict(model='ideal_projective'), duration_ns=0.)])
        _, stages, total = compile_program(self.model, p)
        self.assertEqual(total, 0.)
        self.assertEqual(stages[0]['end_ns'], 0.)

    def test_duration_tolerance_uses_canonical_endpoint(self):
        _, stages, _ = self.compile_measure(self.measurement(duration_ns=300.+5e-10))
        self.assertEqual(stages[0]['end_ns'], 300.)
        self.assertEqual(stages[-1]['start_ns'], 500.)

    def test_short_response_omitted_drive_uses_same_authoritative_default(self):
        op = self.measurement(instrument=dict(model='dispersive_qnd_heterodyne', duration_ns=100., response_order=16))
        _, stages, _ = self.compile_measure(op)
        self.assertEqual(stages[0]['duration_ns'], 100.)
        self.assertEqual(stages[0]['qnd'].metadata['drive_duration_ns'], 100.)
        self.assertEqual(stages[-1]['start_ns'], 300.)

    def test_classified_probabilities_sum_to_unconditional_qnd_map(self):
        cfg = ms.qnd_defaults(['q0']); cfg['response_order'] = 16
        inst, lab = ms.qnd_instrument(self.model, cfg)
        psi = (self.model.encoding[:, 0]+self.model.encoding[:, 2])/np.sqrt(2)
        b = ms.pure_branch(psi, 2)
        all_outcomes, _ = ms.apply_qnd(b, inst, lab, 'classified')
        unrecorded, _ = ms.apply_qnd(b, inst, lab, 'unconditional')
        density = sum(out.density() for _, out in all_outcomes)
        np.testing.assert_allclose(density, unrecorded[0][1].density(), atol=1e-10)
        self.assertAlmostEqual(sum(out.probability() for _, out in all_outcomes), 1., places=11)

    def test_record_trajectory_retains_seed_record_likelihood_and_rejects_multiinput_channel(self):
        cfg = ms.qnd_defaults(['q0']); cfg['response_order'] = 16
        inst, lab = ms.qnd_instrument(self.model, cfg)
        b = ms.pure_branch(self.model.encoding[:, 0], 2)
        sampled, evidence = ms.apply_qnd(b, inst, lab, 'record', seed=42)
        supplied, supplied_evidence = ms.apply_qnd(b, inst, lab, 'record', record=evidence['record'])
        self.assertEqual(evidence['seed'], 42)
        self.assertIsNone(supplied_evidence['seed'])
        self.assertTrue(np.isfinite(evidence['log_likelihood']))
        np.testing.assert_allclose(sampled[0][1].density(), supplied[0][1].density(), atol=1e-12)
        process = ms.process_branch(self.model.encoding[:, :2], 2)
        with mock.patch.object(ms, 'sample_record', side_effect=AssertionError('must not sample')):
            with self.assertRaisesRegex(ValueError, 'not a multi-input channel'):
                ms.apply_qnd(process, inst, lab, 'record', seed=42)

    def test_complete_reordered_encoding_keeps_measurement_labels_and_zz(self):
        labels = ['11', '00', '10', '01']
        model = build_model(demo_cell(), (3, 3, 3), encoding_labels=labels)
        b = ms.pure_branch(model.encoding[:, labels.index('01')], 2)
        outcomes = ms.ideal_outcomes(model, ['q0', 'q1'], basis='dressed_computational')
        measured = dict((key, out.probability()) for key, out in ms.apply_ideal(b, outcomes))
        self.assertAlmostEqual(measured['01'], 1., places=12)
        self.assertAlmostEqual(static_zz_MHz(model)['q0-q1'], static_zz_MHz(self.model)['q0-q1'], places=10)

    def test_local_reset_preserves_trace_residuals_and_removes_correlations(self):
        psi = np.zeros(27, complex)
        psi[0] = psi[np.ravel_multi_index((1, 0, 1), (3, 3, 3))] = 1/np.sqrt(2)
        branch = ms.pure_branch(psi, 2)
        branch.W *= .37
        target = np.diag([.7, .2, .1])
        out = ms.replacement_reset(self.model, branch, modes=['q0'], state=target)
        environment = np.zeros((9, 9), complex); environment[0, 0] = environment[1, 1] = .5
        expected = .37*np.kron(target, environment)
        np.testing.assert_allclose(out.density(), expected, atol=1e-14)
        self.assertAlmostEqual(out.probability(), .37, places=13)
        self.assertEqual(out.discarded_trace, 0.)

    def test_full_reset_is_linear_fixed_replacement(self):
        psi = self.model.encoding[:, 3]
        branch = ms.pure_branch(psi, 2); branch.W *= .31
        target = np.zeros(27, complex); target[2] = 1.
        out = ms.replacement_reset(self.model, branch, kind='full', state=target)
        np.testing.assert_allclose(out.density(), .31*np.outer(target, target.conj()), atol=1e-14)

    def test_capped_reset_rejects_nonclosed_preparation_without_renormalizing(self):
        capped = build_model(demo_cell(), (3, 3, 3), cap=1, encoding_labels=['00'], encoding_kind='bare')
        b = ms.pure_branch(capped.encoding[:, 0], 2)
        with self.assertRaisesRegex(ValueError, 'not closed'):
            ms.replacement_reset(capped, b, modes=['q0'], state={'levels': [1]})
        out = ms.replacement_reset(capped, b, modes=['q0'])
        self.assertAlmostEqual(out.probability(), 1.)

    def test_process_reset_preserves_channel_cross_input_action(self):
        states = np.eye(27, dtype=complex)[:, [0, 10]]
        branch = ms.process_branch(states, 2)
        out = ms.replacement_reset(self.model, branch, modes=['q0'])
        blocks = ms.process_blocks(out)
        mapped_cross = sum(B[:, 0:1]@B[:, 1:2].conj().T for B in blocks)
        np.testing.assert_allclose(mapped_cross, 0., atol=1e-14)
        for k in range(2):
            self.assertAlmostEqual(sum(float(np.vdot(B[:, k], B[:, k]).real) for B in blocks), 1.)

    def test_replacement_rejects_oversized_factorization_before_allocation(self):
        branch = ms.process_branch(np.eye(27, dtype=complex)[:, :2], 2)
        with self.assertRaisesRegex(ValueError, 'before allocation'):
            ms.replacement_reset(self.model, branch, kind='full', max_output_bytes=10)

    def test_process_compression_loss_uses_same_trace_units_as_probability(self):
        eps = 1e-8
        V = np.zeros((27, 4), complex)
        V[0, 0] = V[1, 1] = np.sqrt(1-eps)
        V[2, 2] = V[3, 3] = np.sqrt(eps)
        b = ms.Branch({}, V, np.eye(4), np.zeros(2), block=2)
        out = ms.compress(b, tol=1e-6)
        self.assertAlmostEqual(out.discarded_trace, b.probability()-out.probability(), places=14)
        self.assertAlmostEqual(out.discarded_trace, 2*eps, places=14)

    def test_compression_loss_is_counted_once_when_a_branch_splits(self):
        program = dict(operations=[dict(op='measure', key=key, qubits=['q0'],
                      instrument=dict(model='ideal_projective')) for key in ('first', 'second')])
        original = ms.compress; injected = []
        def with_known_loss(branch, tol):
            result = original(branch, tol)
            if not injected and result.probability() > .9:
                eps = 1e-8; result.V *= np.sqrt(1-eps); result.discarded_trace += eps; injected.append(eps)
            return result
        V = np.zeros((27, 1), complex); V[0, 0] = 1.
        with mock.patch.object(ms, 'compress', side_effect=with_known_loss):
            out = run_program(self.model, program, process_inputs=V)
        self.assertAlmostEqual(out['compression_discarded_trace'], injected[0], places=14)
        self.assertLess(out['probability_balance_error'], 1e-14)

    def test_eleven_modes_selected_encoding_short_fresh_propagation(self):
        cell = linear_chain_cell(6)
        self.assertEqual(len(set(cell['mode_labels'])), 11)
        self.assertEqual(cell['mode_roles'].count('data'), 6)
        self.assertEqual(cell['mode_roles'].count('coupler'), 5)
        self.assertEqual(basis_dimension((3,)*11, 6), 8427)
        model = build_model(cell, (3,)*11, cap=2, encoding_labels=['000000', '100000'], encoding_kind='bare')
        self.assertEqual(model.basis.dimension, 78)
        self.assertEqual(model.encoding.shape, (78, 2))
        schedule = dict(duration_ns=.2, flux=[], drives=[dict(mode='q0', start_ns=0., duration_ns=.2,
            amplitude_GHz=.02, frequency_GHz=4.9, phase_rad=.3, envelope='square')])
        result = propagate_states(model, schedule, model.encoding, rtol=1e-10, atol=1e-12, samples=2)
        self.assertLess(result['norm_error'], 1e-8)
        self.assertGreater(np.linalg.norm(result['final']-model.encoding), 1e-4)

    def test_selected_encoding_rejects_excluded_bitstrings(self):
        with self.assertRaisesRegex(ValueError, 'excludes'):
            build_model(linear_chain_cell(6), (3,)*11, cap=2, encoding_labels=['111111'], encoding_kind='bare')

    def test_default_mode_orders_remain_unchanged(self):
        self.assertEqual(mode_order(2), ['q0', 'c01', 'q1'])
        self.assertEqual(mode_order(3), ['q0', 'c01', 'q1', 'c12', 'q2'])
        self.assertEqual(mode_order(5), ['q0', 'c01', 'q1', 'c12', 'q2', 'c23', 'q3', 'c34', 'q4'])

    def test_split_carrier_clock_matches_uninterrupted(self):
        pulse = dict(mode='q0', start_ns=0., duration_ns=2., amplitude_GHz=.012,
                     frequency_GHz=4.7, phase_rad=.41, envelope='square')
        full = dict(duration_ns=2., flux=[], drives=[pulse])
        half = copy.deepcopy(full); half['duration_ns'] = 1.; half['drives'][0]['duration_ns'] = 1.
        V = self.model.encoding[:, :1]
        whole = propagate_states(self.model, full, V, rtol=1e-11, atol=1e-13, samples=2)
        first = propagate_states(self.model, half, V, rtol=1e-11, atol=1e-13, samples=2)
        second = propagate_states(self.model, half, first['final'], carrier_time_offset_ns=1., rtol=1e-11, atol=1e-13, samples=2)
        np.testing.assert_allclose(second['final'], whole['final'], atol=1e-8, rtol=0)

    def test_program_resume_preserves_clock_branch_weights_and_frames(self):
        pulse = dict(duration_ns=.5, flux=[], drives=[dict(mode='q0', start_ns=0., duration_ns=.5,
            amplitude_GHz=.02, frequency_GHz=4.7, phase_rad=.2, envelope='square')])
        program = dict(feedback_latency_ns=.2, operations=[
            dict(op='evolve', schedule=pulse), dict(op='virtual_z', qubit='q0', angle_rad=.4),
            dict(op='measure', key='m', qubits=['q0'], instrument=dict(model='ideal_projective')),
            dict(op='evolve', schedule=pulse, condition=dict(key='m', equals='1'))])
        whole = run_program(self.model, program)
        checkpoints = []
        part = run_program(self.model, program, checkpoint_callback=checkpoints.append, stop_after_stages=3)
        self.assertFalse(part['complete'])
        self.assertEqual(checkpoints[-1]['clock_ns'], .5)
        resumed = run_program(self.model, program, resume_state=checkpoints[-1])
        self.assertTrue(resumed['complete'])
        self.assertEqual(resumed['duration_ns'], whole['duration_ns'])
        np.testing.assert_allclose(resumed['probabilities'], whole['probabilities'], atol=1e-11)
        for a, b in zip(resumed['branches'], whole['branches']):
            self.assertEqual(a.record, b.record)
            np.testing.assert_allclose(a.frame, b.frame, atol=0.)
            np.testing.assert_allclose(a.density(), b.density(), atol=1e-8)
        broken = dict(checkpoints[-1], clock_ns=0.)
        with self.assertRaisesRegex(ValueError, 'clock'):
            run_program(self.model, program, resume_state=broken)


if __name__ == '__main__':
    unittest.main()
