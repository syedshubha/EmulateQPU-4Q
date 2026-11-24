"""Versioned public contracts. Validation never constructs a Hamiltonian."""
from __future__ import annotations
from copy import deepcopy
from importlib.resources import files
from types import SimpleNamespace
from typing import Any, Literal, NotRequired, TypedDict
import json
import math
from .io import digest

EXPERIMENT_SCHEMA = 'emulateqpu.experiment.v1'
BASIS_ORDER = 'lexicographic local levels; first mode most significant'


class ExperimentResources(TypedDict):
    memory_mb: float
    wall_time_s: float
    numerical_threads: Literal[1]
    max_branches: NotRequired[int]
    max_columns: NotRequired[int]
    max_output_mb: NotRequired[float]


class DrivenStageWork(TypedDict):
    operation_index: int
    operation: Literal['evolve', 'measure']
    conditional: bool
    duration_ns: float
    max_carrier_GHz: float
    max_step_ns: float
    accepted_steps_floor: int
    rhs_evaluations_floor: int


class CarrierWork(TypedDict):
    stages: list[DrivenStageWork]
    unconditional_rhs_evaluations_floor: int
    conditional_rhs_evaluations_floor_if_all_run: int


class ExperimentPlan(TypedDict):
    schema: Literal['emulateqpu.plan.v1']
    request_sha256: str
    engine: Literal['chain', 'cavity']
    dimension: int
    local_dimensions: list[int]
    mode_counts: dict[str, int]
    state_columns: int
    branches_upper_bound: int
    workers: int
    numerical_threads: Literal[1]
    estimated_peak_memory_bytes: int
    per_worker_memory_bytes: int
    aggregate_memory_bytes: int
    wall_time_s: float
    memory_budget_bytes: int
    estimates: dict[str, Any]
    carrier_work: NotRequired[CarrierWork | None]
    qualification: str


class StudyTask(TypedDict):
    id: str
    request: dict[str, Any]


class Study(TypedDict):
    schema: Literal['emulateqpu.study.v1']
    purpose: NotRequired[str]
    tasks: list[StudyTask]


Backend = Literal['serial', 'local-process', 'slurm']


class ExecutionPlan(TypedDict):
    schema: Literal['emulateqpu.execution_plan.v1']
    study: Study
    study_sha256: str
    backend: Backend
    workers: int
    numerical_threads_per_worker: Literal[1]
    task_count: int
    task_plans: list[ExperimentPlan]
    per_task_estimated_peak_bytes: list[int]
    coordinator_memory_bytes: int
    aggregate_estimated_peak_bytes: int
    aggregate_memory_limit_bytes: int | None
    estimates_are_bounds: Literal[False]
    accepted: Literal[True]


def memory_bytes(value: float, *, field: str) -> int:
    """Convert a finite positive MiB declaration to whole bytes."""
    if type(value) not in (int, float):
        raise ValueError(f'{field} must be a finite positive number of MiB')
    scaled = value * 1024**2
    if (isinstance(scaled, float) and not math.isfinite(scaled)) or scaled < 1:
        raise ValueError(f'{field} must specify at least one byte and be finite')
    return int(scaled)


def resource_limits(resources: ExperimentResources) -> tuple[int, float]:
    """Return the worker's authoritative memory and wall-time limits."""
    wall_time = resources['wall_time_s']
    if type(wall_time) not in (int, float):
        raise ValueError('wall_time_s must be a finite positive number')
    try:
        wall_time = float(wall_time)
    except OverflowError as exc:
        raise ValueError('wall_time_s must be a finite positive number') from exc
    if not math.isfinite(wall_time) or wall_time <= 0:
        raise ValueError('wall_time_s must be a finite positive number')
    return memory_bytes(resources['memory_mb'], field='memory_mb'), wall_time


def aggregate_limit_bytes(profile: dict[str, Any] | None) -> int | None:
    if profile is None:
        return None
    if not isinstance(profile, dict):
        raise ValueError('Resource profile must be an object')
    if 'aggregate_memory_mb' not in profile:
        return None
    return memory_bytes(profile['aggregate_memory_mb'], field='aggregate_memory_mb')


def schema(name):
    return json.loads(files('emulateqpu').joinpath('schemas', name+'.v1.json').read_text())


def calibration_binding(request):
    """Circuit, extraction, junction, basis, line and port identity, without fitting."""
    return digest({'device': request['device'], 'model': request['model']})


def bind_calibration(request, *, source, achieved_metrics):
    r = deepcopy(request)
    r['calibration'] = {'binding_sha256': calibration_binding(r), 'schedule_sha256': digest(r.get('schedule', r.get('program'))),
                        'source': source, 'achieved_metrics': deepcopy(achieved_metrics)}
    return r


def copy_uncalibrated(request):
    r = deepcopy(request); r['calibration'] = None
    r['purpose'] = 'Copied editable waveform; uncalibrated on this model'
    return r


def _labels(labels, nd, name):
    if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
        raise ValueError(name+' requires unique bitstrings')
    if any(not isinstance(b, str) or len(b) != nd or set(b)-{'0','1'} for b in labels):
        raise ValueError(name+' requires one bit per data mode')


def validate_request(request):
    import jsonschema
    from emulateqpu._engine.model import validate_cell
    from emulateqpu._engine.evolve import validate_schedule_v2
    r = deepcopy(request)
    try: jsonschema.Draft202012Validator(schema('experiment')).validate(r)
    except jsonschema.ValidationError as e: raise ValueError(e.message) from e
    # Also rejects nonfinite values that Python JSON would otherwise accept.
    digest(r)
    if r.get('noise') is not None: raise ValueError('Chain noise is not implemented; no substitution is made')
    if r['engine'] == 'cavity':
        if set(r)-{'schema','engine','cavity','resources','name','seed','purpose'}:
            raise ValueError('Cavity requests use the cavity specification, not chain fields')
        from .cavity import validate_request as cv
        inner=cv(r['cavity'])
        if r['resources']['memory_mb']*1024**2!=inner['resources']['memory_bytes'] or r['resources']['wall_time_s']!=inner['resources']['wall_time_s']:
            raise ValueError('Cavity inner and execution memory/wall budgets must agree')
        return r
    required = {'device','model','initial','solver','observables'}
    if required-set(r): raise ValueError('Missing chain fields: '+str(sorted(required-set(r))))
    if ('schedule' in r) == ('program' in r): raise ValueError('Choose exactly one schedule or program')
    c = validate_cell(r['device']['cell'])
    # Do not normalize the cell: hashes refer to the actual supplied document.
    nd = c['mode_roles'].count('data'); dims = r['model']['dimensions']
    if len(dims) != len(c['mode_labels']): raise ValueError('One local dimension per mode required')
    if any(d > 2*c['ncut']+1 for d in dims): raise ValueError('Local dimension exceeds the finite charge basis')
    if any(dims[i] < 3 for i,role in enumerate(c['mode_roles']) if role == 'data'):
        raise ValueError('Data modes need three levels to represent leakage')
    enc = r['model']['encoding']; _labels(enc['labels'], nd, 'encoding')
    initial = r['initial']
    if initial.get('kind') != 'encoding' or set(initial)-{'kind','labels'}:
        raise ValueError('Initial state selects explicitly labelled encoding columns')
    _labels(initial.get('labels'), nd, 'initial')
    if set(initial['labels'])-set(enc['labels']): raise ValueError('Initial labels absent from declared encoding')
    labels=set(c['mode_labels'])
    for edge in r['device']['topology']:
        if not isinstance(edge,list) or len(edge)!=2 or len(set(edge))!=2 or set(edge)-labels:
            raise ValueError('Topology edges must join two distinct mode labels')
    light = SimpleNamespace(cell=c, data_modes=tuple(i for i,x in enumerate(c['mode_roles']) if x=='data'),
                            coupler_modes=tuple(i for i,x in enumerate(c['mode_roles']) if x=='coupler'))
    from emulateqpu._engine.program import measurement_interval_ns
    def check_measurement(op):
        measurement_interval_ns(op)
        qs=op.get('qubits')
        data_labels=[q for q,role in zip(c['mode_labels'],c['mode_roles']) if role=='data']
        if not isinstance(qs,list) or not qs or len(qs)!=len(set(qs)) or set(qs)-set(data_labels): raise ValueError('Measurement needs unique data labels')
        inst=op.get('instrument',{})
        if inst.get('model','ideal_projective') not in ('ideal_projective','dispersive_qnd_heterodyne'): raise ValueError('Unsupported measurement model')
        if 'channels' in inst and (not isinstance(inst['channels'],list) or not 1<=len(inst['channels'])<=9): raise ValueError('Effective QND supports one to nine declared channels')
        if inst.get('outcome','classified') not in ('unconditional','classified','record'): raise ValueError('Unsupported QND outcome mode')
        if inst.get('outcome')=='record' and (len(initial['labels'])!=1 or r.get('metrics')): raise ValueError('Continuous record is one selected trajectory, not an unconditional multi-input channel')
        if op.get('concurrent') is not None: validate_schedule_v2(light,op['concurrent'])
    if r.get('measurement') is not None: check_measurement(r['measurement'])
    if 'schedule' in r: validate_schedule_v2(light, r['schedule'])
    if 'program' in r:
        p=r['program']
        if not isinstance(p.get('operations'),list) or not 1 <= len(p['operations']) <=64:
            raise ValueError('Program needs 1..64 operations')
        if p.get('initial') is not None: raise ValueError('Use request.initial for the program input')
        if set(p)-{'schema','operations','feedback_latency_ns','max_branches','compress_tol','prune_probability','regions','solver','purpose'}: raise ValueError('Unknown program field')
        if 'solver' in p and p['solver']!={k:r['solver'][k] for k in ('rtol','atol','samples')}: raise ValueError('Declare one authoritative solver in request.solver')
        if 'max_branches' in p and p['max_branches']!=r['resources'].get('max_branches',64): raise ValueError('Program and resource branch limits must agree')
        known_keys=set();prior_branching=False
        for op in p['operations']:
            if not isinstance(op,dict): raise ValueError('Program operation must be an object')
            allowed={'evolve':{'op','schedule','condition'},'wait':{'op','duration_ns','condition'},
                     'measure':{'op','key','qubits','instrument','duration_ns','concurrent'},
                     'reset':{'op','model','modes','state','condition','max_output_bytes'},
                     'virtual_z':{'op','qubit','angle_rad','condition'},'observe':{'op','label'}}
            if set(op)-allowed.get(op.get('op'),set()): raise ValueError('Unknown operation field or operation')
            condition=op.get('condition')
            if condition is not None and (set(condition)!={'key','equals'} or condition['key'] not in known_keys): raise ValueError('Condition requires an earlier reported measurement')
            if op.get('op')=='measure':
                instrument=op.get('instrument',{})
                if instrument.get('outcome')=='record' and prior_branching: raise ValueError('Continuous record requires one unbranched trajectory')
                if instrument.get('model','ideal_projective')=='ideal_projective' or instrument.get('outcome','classified')=='classified': prior_branching=True
                if not isinstance(op.get('key'),str) or not op['key'] or op['key'] in known_keys: raise ValueError('Unique measurement key required')
                known_keys.add(op['key'])
            if op.get('op')=='virtual_z' and op.get('qubit') not in [q for q,role in zip(c['mode_labels'],c['mode_roles']) if role=='data']: raise ValueError('Virtual Z requires a data label')
            if op.get('op')=='reset' and op.get('model') not in ('local_replacement','fixed_state_replacement'): raise ValueError('Unsupported reset channel')
            if op.get('op') not in ('evolve','wait','measure','reset','virtual_z','observe'): raise ValueError('Unsupported program operation')
            if op.get('op') == 'evolve': validate_schedule_v2(light, op['schedule'])
            if op.get('op') == 'measure': check_measurement(op)
        if r.get('measurement') is not None: raise ValueError('Declare measurements inside the program')
    if r['solver']['rtol'] < 1e-13 or r['solver']['rtol'] > 1e-6 or r['solver']['atol'] < 1e-15 or r['solver']['atol'] >1e-8:
        raise ValueError('Solver tolerances outside supported engine ranges')
    cal = r.get('calibration')
    if cal is not None:
        if cal.get('binding_sha256') != calibration_binding(r):
            raise ValueError('Calibration binding invalidated; explicitly copy an uncalibrated waveform')
        if cal.get('schedule_sha256') != digest(r.get('schedule', r.get('program'))):
            raise ValueError('Calibrated waveform changed; explicitly copy an uncalibrated waveform')
    metrics=r.get('metrics')
    if metrics:
        if set(metrics)-{'target_unitary','output_phases_rad'}: raise ValueError('Unknown metric request')
        if enc['kind'] != 'dressed' or len(enc['labels']) != 2**nd or set(initial['labels']) != set(enc['labels']):
            raise ValueError('Unconditional gate fidelity requires every dressed computational input')
        import numpy as np
        target=metrics.get('target_unitary',{});U=np.asarray(target.get('real'),float)+1j*np.asarray(target.get('imag'),float)
        phases=np.asarray(metrics.get('output_phases_rad'),float);d=2**nd
        if U.shape!=(d,d) or phases.shape!=(nd,) or not np.isfinite(U).all() or not np.isfinite(phases).all() or np.max(abs(U.conj().T@U-np.eye(d)))>1e-10:
            raise ValueError('Gate metric requires finite matching unitary and explicit output frames')
    return r


def example_request(name='small-pulse'):
    if name == 'cavity':
        from .cavity import example_request as ce
        return {'schema':EXPERIMENT_SCHEMA,'engine':'cavity','cavity':ce(),
                'resources':{'memory_mb':512,'wall_time_s':60,'numerical_threads':1}}
    if name not in ('small-pulse','feedback','four-qubit','six-data'): raise ValueError('Unknown example: '+name)
    from emulateqpu._engine.fixtures import demo_cell, demo_schedule
    c=demo_cell(); dims=[3,3,3]; cap=None; enc=['00','01','10','11']; kind='dressed'; initial=['00']
    s=demo_schedule()
    measured_qubits = ['q0', 'q1', 'q2', 'q3'] if name == 'four-qubit' else ['q0']
    if name in ('four-qubit', 'six-data'):
        from .chain import linear_chain_cell
        n_data = 4 if name == 'four-qubit' else 6
        c = linear_chain_cell(n_data)
        dims = [3] * (2 * n_data - 1)
        cap = 2
        enc = ['0' * n_data, '1' + '0' * (n_data - 1)]
        initial = [enc[1]]
        kind = 'bare'
        s={'duration_ns':.04,'drives':[{'mode':'q0','start_ns':0.,'duration_ns':.04,'amplitude_GHz':.01,
                                      'frequency_GHz':5.,'phase_rad':.3,'envelope':'square'}],'flux':[]}
    r={'schema':EXPERIMENT_SCHEMA,'name':name,'engine':'chain',
       'device':{'schema':'emulateqpu.device.v1','cell':c,'topology':[[a,b] for a,b in zip(c['mode_labels'][:-1],c['mode_labels'][1:])],
                 'geometry':None,'extraction_identity':None,'parameter_origin':'assumed'},
       'model':{'dimensions':dims,'excitation_cap':cap,'cap_exempt':[],
                'encoding':{'kind':kind,'labels':enc},'basis_order':BASIS_ORDER},
       'initial':{'kind':'encoding','labels':initial},'schedule':s,
       'measurement':{'qubits':measured_qubits,'instrument':{'model':'ideal_projective','basis':'local_levels'},'duration_ns':0.},
       'noise':None,'solver':{'rtol':1e-10,'atol':1e-12,'samples':3},
       'resources':{'memory_mb':512,'wall_time_s':120,'numerical_threads':1,'max_branches':64,'max_output_mb':64},
       'observables':['populations','measurement','states'],'metrics':None,'calibration':None,'seed':1729,
       'purpose':'Editable uncalibrated finite-model software example; no spatial or physical qualification'}
    if name == 'feedback':
        r.pop('schedule');r['measurement']=None
        r['program']={'feedback_latency_ns':.1,'operations':[{'op':'evolve','schedule':s},
           {'op':'measure','key':'m','qubits':['q0'],'instrument':{'model':'ideal_projective'},'duration_ns':0.},
           {'op':'evolve','condition':{'key':'m','equals':'1'},'schedule':s}]}
    return r
