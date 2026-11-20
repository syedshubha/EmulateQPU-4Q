"""Public contract/installation-independent integration risks on tiny finite models."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from emulateqpu import example,run_request
from emulateqpu.api import load_model,load_result
from emulateqpu.contracts import validate_request,bind_calibration,copy_uncalibrated
from emulateqpu.planner import basis_dimension,plan_request
from emulateqpu.io import inspect_result


class Application(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_fresh_changed_waveform_measurement_and_reload(self):
        r=example();a=run_request(r,self.root/'a')
        r['schedule']['drives'][0]['amplitude_GHz']=0.
        b=run_request(r,self.root/'b')
        self.assertEqual(a['status'],'PASS');self.assertEqual(b['status'],'PASS')
        self.assertGreater(abs(a['summary']['bare_level_populations']['q0'][1]-b['summary']['bare_level_populations']['q0'][1]),1e-3)
        loaded,branches=load_result(self.root/'a')
        self.assertEqual(loaded,a);self.assertAlmostEqual(sum(x.probability() for x in branches),1.,places=8)
        m=load_model(self.root/'a');self.assertEqual(m.identity(),a['model_identity'])
        with (self.root/'a'/'states.npz').open('ab') as f:f.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'integrity'):inspect_result(self.root/'a')

    def test_calibration_binding_and_explicit_uncalibrated_copy(self):
        r=bind_calibration(example(),source='test only',achieved_metrics={'scope':'synthetic fixture'})
        validate_request(r)
        for field in ('EC_matrix_GHz','EJ_idle_GHz','lines'):
            altered=deepcopy(r)
            if field=='EC_matrix_GHz':altered['device']['cell'][field][0][0]+=.001
            elif field=='EJ_idle_GHz':altered['device']['cell'][field][0]+=.1
            else:altered['device']['cell'][field]['c01']['gain']=.99
            with self.assertRaisesRegex(ValueError,'binding'):validate_request(altered)
            validate_request(copy_uncalibrated(altered))
        altered=deepcopy(r);altered['device']['geometry']='changed'
        with self.assertRaisesRegex(ValueError,'binding'):validate_request(altered)
        altered=deepcopy(r);altered['model']['dimensions']=[4,3,3]
        with self.assertRaisesRegex(ValueError,'binding'):validate_request(altered)
        altered=deepcopy(r);altered['schedule']['drives'][0]['amplitude_GHz']*=.5
        with self.assertRaisesRegex(ValueError,'waveform'):validate_request(altered)

    def test_checkpoint_resume_matches_complete_without_eigensolve(self):
        r=example('feedback')
        full=run_request(r,self.root/'full')
        partial=run_request(r,self.root/'partial',stop_after_stages=2)
        self.assertFalse(partial['complete'])
        with patch('emulateqpu._engine.model.build_model',side_effect=AssertionError('resume must not rebuild')):
            resumed=run_request(r,self.root/'resumed',resume_from=self.root/'partial')
        self.assertEqual(full['duration_ns'],resumed['duration_ns'])
        _,a=load_result(self.root/'full');_,b=load_result(self.root/'resumed')
        self.assertEqual([x.record for x in a],[x.record for x in b])
        for x,y in zip(a,b):
            self.assertLess(np.max(abs(x.density()-y.density())),1e-9)
            np.testing.assert_array_equal(x.frame,y.frame)
        changed=deepcopy(r);changed['seed']+=1
        with self.assertRaisesRegex(ValueError,'request changed'):run_request(changed,self.root/'bad',resume_from=self.root/'partial')

    def test_resource_rejection_precedes_construction(self):
        self.assertEqual(basis_dimension([3]*11,6),8427)
        r=example('six-data');p=plan_request(r)
        self.assertEqual(p['dimension'],78);self.assertEqual(p['mode_counts'],{'data':6,'coupler':5,'cavity':0})
        r['resources']['memory_mb']=1
        with patch('emulateqpu._engine.model.build_model',side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(ValueError,'Preallocation'):run_request(r,self.root/'rejected')
        self.assertFalse((self.root/'rejected').exists())

    def test_eleven_mode_public_execution(self):
        r=run_request(example('six-data'),self.root/'six')
        self.assertTrue(r['complete']);self.assertEqual(r['actual_model']['dimension'],78)
        self.assertEqual(len(r['actual_model']['mode_labels']),11)
        self.assertIsNone(r['summary']['mean_dressed_computational_leakage'])
        self.assertFalse(r['scientific_qualification']['numerical_convergence'])
        self.assertLess(r['checks']['probability_error'],1e-8)

    def test_unsupported_requests_rejected_without_model(self):
        cases=[]
        r=example();r['noise']={'T1_us':20};cases.append(r)
        r=example();r['measurement']['instrument']['model']='nonlinear_receiver';cases.append(r)
        r=example('feedback');r['program']['operations'][0]['op']='ideal_gate';cases.append(r)
        r=example();r['initial']['labels']=['00','01'];r['measurement']['instrument']={'model':'dispersive_qnd_heterodyne','outcome':'record'};r['measurement'].pop('duration_ns');cases.append(r)
        for r in cases:
            with self.assertRaises(ValueError):validate_request(r)

    def test_metrics_all_inputs_and_explicit_frames(self):
        r=example();r['measurement']=None;r['schedule']={'duration_ns':0.,'drives':[],'flux':[]}
        r['initial']['labels']=['00','01','10','11'];r['metrics']={'target_unitary':{'real':np.eye(4).tolist(),'imag':np.zeros((4,4)).tolist()},'output_phases_rad':[0.,0.]}
        result=run_request(r,self.root/'gate')
        self.assertAlmostEqual(result['summary']['metrics']['unconditional_gate_fidelity'],1.,places=10)
        r['initial']['labels']=['00']
        with self.assertRaisesRegex(ValueError,'every dressed'):validate_request(r)

    def test_program_virtual_frame_in_unconditional_metric(self):
        r=example();r.pop('schedule');r['measurement']=None
        r['initial']['labels']=['00','01','10','11']
        r['program']={'operations':[{'op':'virtual_z','qubit':'q0','angle_rad':np.pi/2}]}
        U=np.diag([1,1,1j,1j]);r['metrics']={'target_unitary':{'real':U.real.tolist(),'imag':U.imag.tolist()},'output_phases_rad':[0.,0.]}
        result=run_request(r,self.root/'frame')
        self.assertAlmostEqual(result['summary']['metrics']['unconditional_gate_fidelity'],1.,places=10)

    def test_record_rejects_prior_branching_and_seeds_are_event_bound(self):
        from emulateqpu.api import _program
        r=example('feedback');r['program']['operations']=[
            {'op':'measure','key':'a','qubits':['q0'],'instrument':{'model':'ideal_projective'}},
            {'op':'measure','key':'b','qubits':['q0'],'instrument':{'model':'dispersive_qnd_heterodyne','outcome':'record'}}]
        with self.assertRaisesRegex(ValueError,'unbranched'):validate_request(r)
        r['program']['operations'][0]['instrument']={'model':'dispersive_qnd_heterodyne','outcome':'unconditional'}
        validate_request(r);p=_program(r)
        self.assertNotEqual(p['operations'][0]['instrument']['seed'],p['operations'][1]['instrument']['seed'])
        self.assertEqual(p,_program(r))

    def test_qnd_label_table_is_budgeted_separately_from_excitation_cap(self):
        from emulateqpu.chain import linear_chain_cell
        r=example('six-data');r['device']['cell']=linear_chain_cell(20);r['device']['topology']=[]
        r['model'].update(dimensions=[3]*39,excitation_cap=0,encoding={'kind':'bare','labels':['0'*20]})
        r['initial']['labels']=['0'*20];r['schedule']={'duration_ns':0.,'drives':[],'flux':[]}
        r['measurement']={'qubits':['q0'],'instrument':{'model':'dispersive_qnd_heterodyne','outcome':'unconditional','channels':[{'reads':'q0','chi_MHz':{'q'+str(i):1. for i in range(20)}}]}}
        with self.assertRaisesRegex(ValueError,'Preallocation'):plan_request(r)

    def test_schema_roundtrip(self):
        import jsonschema
        from emulateqpu.contracts import schema
        r=example();jsonschema.validate(r,schema('experiment'));jsonschema.validate(r['device'],schema('device'))
        result=run_request(r,self.root/'schema')
        jsonschema.validate(result,schema('result'));jsonschema.validate(plan_request(r),schema('plan'))

if __name__=='__main__':unittest.main()
