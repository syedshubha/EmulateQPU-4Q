"""One request/result API for fresh simulations and operation-boundary resume."""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
import os
import tempfile
import time
import numpy as np
from threadpoolctl import threadpool_limits
from . import __version__
from ._resources import WallTimeLimitError, process_peak_rss_bytes
from .contracts import validate_request
from .planner import _plan_validated_request
from .io import atomic_json, digest, file_hash, read_json, inspect_result


@lru_cache(maxsize=1)
def source_identity():
    """Hash the installed supported code, including the authoritative engine."""
    rows={}
    for package in ('emulateqpu','emulateqpu._engine'):
        root=files(package)
        for p in sorted(root.iterdir(),key=lambda x:x.name):
            if p.name.endswith('.py'): rows[package+'/'+p.name]=__import__('hashlib').sha256(p.read_bytes()).hexdigest()
    bath='port_bath.py'
    rows['emulateqpu._physics/'+bath]=__import__('hashlib').sha256(files('emulateqpu._physics').joinpath(bath).read_bytes()).hexdigest()
    return {'package_version':__version__,'source_sha256':digest(rows),'source_files':rows,
            'numpy_version':np.__version__,'scipy_version':__import__('scipy').__version__}


def _npz_atomic(path, arrays):
    path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _encode_branches(branches):
    arrays={};meta=[]
    for i,b in enumerate(branches):
        for key in ('V','W','frame'): arrays[f'b{i}_{key}']=getattr(b,key)
        meta.append({'record':b.record,'discarded_trace':b.discarded_trace,'log':b.log,'block':b.block})
    return meta,arrays


def _decode_branches(meta, arrays):
    from emulateqpu._engine.measurement import Branch
    return [Branch(m['record'],arrays[f'b{i}_V'].copy(),arrays[f'b{i}_W'].copy(),arrays[f'b{i}_frame'].copy(),
                   m['discarded_trace'],m['log'],m['block']) for i,m in enumerate(meta)]


def export_model(model, output):
    """Export finite operators without requiring a fresh eigensolve to reload."""
    out=Path(output); arrays={}
    for k in ('states','codes','strides','parity'): arrays['basis_'+k]=getattr(model.basis,k)
    arrays.update(encoding=model.encoding,energies=model.encoded_energies,comp_rows=model.comp_rows)
    operators={'H':model.H,**{'cos_'+str(i):v for i,v in model.cos_ops.items()},
               **{'charge_'+str(i):v for i,v in model.charge_ops.items()}}
    for key,A in operators.items():
        arrays[key+'_data']=A.data;arrays[key+'_indices']=A.indices;arrays[key+'_indptr']=A.indptr
    for i,local in enumerate(model.local):
        for j,A in enumerate(local): arrays[f'local_{i}_{j}']=A
    _npz_atomic(out/'model.npz',arrays)
    meta={'schema':'emulateqpu.model_export.v1','cell':model.cell,'dimensions':list(model.basis.dims),
          'cap':model.basis.cap,'weights':model.basis.weights,'data_modes':model.data_modes,
          'coupler_modes':model.coupler_modes,'analysis':model.analysis,'identity':model.identity(),
          'operators':list(operators),'payload_sha256':file_hash(out/'model.npz')}
    atomic_json(out/'model.json',meta);return meta


def load_model(output):
    from scipy.sparse import csr_matrix
    from emulateqpu._engine.model import Basis,ChainModel
    out=Path(output);meta=read_json(out/'model.json')
    if meta['schema']!='emulateqpu.model_export.v1' or file_hash(out/'model.npz')!=meta['payload_sha256']:
        raise ValueError('Model export integrity mismatch')
    with np.load(out/'model.npz',allow_pickle=False) as a:
        D=len(a['basis_codes']);basis=Basis(tuple(meta['dimensions']),meta['cap'],a['basis_states'].copy(),
            a['basis_codes'].copy(),a['basis_strides'].copy(),a['basis_parity'].copy(),
            None if meta['weights'] is None else tuple(meta['weights']))
        ops={k:csr_matrix((a[k+'_data'].copy(),a[k+'_indices'].copy(),a[k+'_indptr'].copy()),shape=(D,D)) for k in meta['operators']}
        local=[tuple(a[f'local_{i}_{j}'].copy() for j in range(3)) for i in range(len(meta['dimensions']))]
        m=ChainModel(meta['cell'],basis,local,ops['H'],{i:ops['cos_'+str(i)] for i in meta['coupler_modes']},
                     {i:ops['charge_'+str(i)] for i in range(len(meta['dimensions']))},tuple(meta['data_modes']),
                     tuple(meta['coupler_modes']),a['comp_rows'].copy(),a['encoding'].copy(),a['energies'].copy(),meta['analysis'])
    if m.identity()!=meta['identity']: raise ValueError('Reloaded finite model identity mismatch')
    return m


def _save_checkpoint(out,state,request_id):
    cp=out/'checkpoints';cp.mkdir(exist_ok=True)
    name='stage-%04d'%state['next_stage'];payload=cp/(name+'.npz');record=cp/(name+'.json')
    if payload.exists() or record.exists(): raise ValueError('Checkpoint would overwrite existing evidence')
    meta,arrays=_encode_branches(state['branches']);_npz_atomic(payload,arrays)
    data={k:v for k,v in state.items() if k!='branches'}
    data.update(branch_metadata=meta,request_sha256=request_id,source_identity=source_identity(),
                payload=payload.name,payload_sha256=file_hash(payload))
    atomic_json(record,data);atomic_json(out/'checkpoint.json',{'record':record.relative_to(out).as_posix(),
                                                             'record_sha256':file_hash(record)})


def _load_checkpoint(run,request_id):
    run=Path(run)
    pointer=run/'checkpoint.json'
    if pointer.is_symlink(): raise ValueError('Checkpoint pointer cannot be a symlink')
    ptr=read_json(pointer);p=run/ptr['record']
    if p.is_symlink() or not p.resolve().is_relative_to(run.resolve()) or file_hash(p)!=ptr['record_sha256']: raise ValueError('Checkpoint manifest integrity mismatch')
    cp=read_json(p)
    if cp['source_identity']!=source_identity() or cp['request_sha256']!=request_id:
        raise ValueError('Checkpoint source, dependency or input identity changed')
    payload=p.parent/cp['payload']
    if payload.is_symlink() or payload.resolve().parent!=p.resolve().parent or file_hash(payload)!=cp['payload_sha256']: raise ValueError('Checkpoint payload integrity mismatch')
    with np.load(payload,allow_pickle=False) as a: cp['branches']=_decode_branches(cp.pop('branch_metadata'),a)
    return cp


def _program(request):
    r=request
    if 'program' in r: p=deepcopy(r['program'])
    else:
        p={'operations':[{'op':'evolve','schedule':deepcopy(r['schedule'])}]}
        if r.get('measurement') is not None: p['operations'].append({'op':'measure','key':'measurement',**deepcopy(r['measurement'])})
    p['max_branches']=r['resources'].get('max_branches',64)
    p['solver']={k:r['solver'][k] for k in ('rtol','atol','samples')}
    for event_index,op in enumerate(p['operations']):
        if op.get('op')=='measure' and op.get('instrument',{}).get('model')=='dispersive_qnd_heterodyne':
            op['instrument'].setdefault('seed',int(digest({'seed':r.get('seed',0),'measurement_event':event_index})[:8],16))
    return p


def _summary(model, result, request):
    probs=result['probabilities'];total=result['input_trace'];nd=len(model.data_modes)
    diagonal=sum((b.diagonal() for b in result['branches']),np.zeros(model.basis.dimension))
    bare={label:np.bincount(model.basis.states[:,i],weights=diagonal,minlength=model.basis.dims[i]).tolist()
          for i,label in enumerate(model.cell['mode_labels'])}
    higher=np.any(model.basis.states[:,model.data_modes]>=2,axis=1)
    bare_leak=float(diagonal[higher].sum())
    enc=request['model']['encoding'];full=(enc['kind']=='dressed' and len(enc['labels'])==2**nd)
    survival=sum(float(np.real(np.trace(b.W@((model.encoding.conj().T@b.V).conj().T@(model.encoding.conj().T@b.V))))) for b in result['branches'])
    metrics={'unconditional_gate_fidelity':None,'state_overlap':None,'unavailable_reason':'No gate target requested'}
    if request.get('metrics'):
        labels=enc['labels'];initial=request['initial']['labels'];order=[initial.index(x) for x in labels]
        spec=request['metrics'];target=spec['target_unitary'];U=np.asarray(target['real'])+1j*np.asarray(target['imag'])
        d=len(labels);ph=np.asarray(spec['output_phases_rad']);bits=np.array([list(map(int,b)) for b in labels]);Z=np.exp(-1j*bits@ph)
        if U.shape!=(d,d) or ph.shape!=(nd,) or np.max(abs(U.conj().T@U-np.eye(d)))>1e-10: raise ValueError('Invalid unitary target/frame')
        terms=0.
        from emulateqpu._engine.measurement import process_blocks
        for b in result['branches']:
            for B in process_blocks(b):
                A=(Z*np.exp(-1j*(bits@b.frame)))[:,None]*(model.encoding.conj().T@B[:,order]);terms+=abs(np.trace(U.conj().T@A))**2+np.sum(abs(A)**2)
        metrics={'unconditional_gate_fidelity':float(terms/(d*(d+1))),'state_overlap':None,'unavailable_reason':None}
    return {'branch_probabilities':probs,'measurement_records':result['records'],'input_trace':total,
            'probability_total':result['probability_total'],'bare_level_populations':bare,
            'mean_bare_data_higher_level_population':bare_leak/total,
            'mean_dressed_computational_leakage':float((total-survival)/total) if full else None,
            'selected_encoding_survival':survival/total,
            'sink_population':0.,'compression_discarded_trace':result['compression_discarded_trace'],
            'pruned_probability':result['pruned_probability'],'metrics':metrics}


def run_request(request, output, *, resume_from=None, stop_after_stages=None):
    """Execute a planned request synchronously, saving immutable artifacts.

    Checkpoints are at complete program operations. A killed in-flight operation
    restarts from its prior boundary; carrier time and branches stay explicit.
    Use the execution backends for a subprocess wall-time/memory watchdog.
    """
    return _run_request(request, output, resume_from=resume_from, stop_after_stages=stop_after_stages)


def _run_request(request, output, *, resume_from=None, stop_after_stages=None, deadline=None):
    r = validate_request(request)
    plan = _plan_validated_request(r)
    out = Path(output).expanduser().resolve()
    if out.exists(): raise ValueError('Use a fresh result directory')
    if r['engine']=='cavity':
        if resume_from is not None or stop_after_stages is not None: raise ValueError('Cavity resume is not supported')
        from .cavity import run_request as run_cavity
        result=run_cavity(r['cavity'],out,deadline=deadline)
        # The outer request remains the execution identity; the adapter spec is retained separately.
        atomic_json(out/'experiment.json',r)
        result.update(request_sha256=digest(r),complete=result.get('complete',True),source_identity=source_identity())
        result.setdefault('scientific_qualification',{'numerical_convergence':False,'physical_validation':False})
        result['status']='PASS' if all(v.get('outcome')=='PASS' for v in result['checks'].values()) else 'FAIL'
        result['payload_sha256']=dict(result.get('artifacts',{}))
        result['payload_sha256']['experiment.json']=file_hash(out/'experiment.json')
        atomic_json(out/'result.json',result);return result
    from emulateqpu._engine.model import build_model
    from emulateqpu._engine.program import run_program
    from emulateqpu._engine.program import encoding_labels
    started=time.monotonic()
    if deadline is None: deadline=started+r['resources']['wall_time_s']
    if started>=deadline: raise WallTimeLimitError('Wall budget reached before simulation')
    out.mkdir(parents=True)
    atomic_json(out/'request.json',r);atomic_json(out/'plan.json',plan)
    identity=digest(r)
    def checkpoint(state):
        _save_checkpoint(out,state,identity)
        if time.monotonic()>=deadline:
            raise WallTimeLimitError('Wall budget reached at operation boundary; checkpoint preserved')
    with threadpool_limits(limits=1):
        m=r['model']
        resume=None
        if resume_from is not None:
            source=Path(resume_from).expanduser().resolve()
            try:
                if digest(read_json(source/'request.json'))!=identity: raise ValueError('Resume request changed')
                resume=_load_checkpoint(source,identity)
                model=load_model(source)
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f'Cannot resume checkpoint from {source}: {error}') from error
        else:
            model=build_model(r['device']['cell'],tuple(m['dimensions']),m['excitation_cap'],
                encoding_method='auto',cap_exempt=tuple(m.get('cap_exempt',[])),
                encoding_labels=m['encoding']['labels'],encoding_kind=m['encoding']['kind'])
        export_model(model,out)
        labels=encoding_labels(model);V=model.encoding[:,[labels.index(x) for x in r['initial']['labels']]]
        if time.monotonic()>=deadline: raise WallTimeLimitError('Wall budget reached before propagation')
        result=run_program(model,_program(r),process_inputs=V,checkpoint_callback=checkpoint,
                           resume_state=resume,stop_after_stages=stop_after_stages)
        if resume is not None and result['next_stage']==resume['next_stage']:
            checkpoint(resume)
    meta,arrays=_encode_branches(result['branches']);_npz_atomic(out/'states.npz',arrays)
    summary=_summary(model,result,r);complete=result['complete']
    observed=process_peak_rss_bytes()
    operation_checks=[x.get('evolution',[]) for x in result['trace']]
    norm_errors=[float(item.get('norm_error',0.)) for group in operation_checks for item in group]
    max_norm_error=max(norm_errors,default=0.)
    checks={'probability_error':result['probability_error'],
            'probability_balance_error':result.get('probability_balance_error',result['probability_error']),
            'probability_conserved':result.get('probability_balance_error',result['probability_error'])<1e-7,
            'maximum_propagation_norm_Gram_error':max_norm_error,
            'propagation_norm_Gram_tolerance':1e-7,
            'propagation_norm_Gram_pass':bool(np.isfinite(norm_errors).all() and max_norm_error<=1e-7),
            'encoding_gram_error':model.analysis.get('encoding_gram_error'),
            'operation_checks':operation_checks}
    ok=checks['probability_conserved'] and checks['propagation_norm_Gram_pass']
    public={'schema':'emulateqpu.result.v1','status':('PASS' if ok else 'FAIL') if complete else 'CHECKPOINTED',
            'complete':complete,'request_sha256':identity,'input_sha256':identity,
            'model_sha256':digest(model.identity()),'model_identity':model.identity(),
            'source_identity':source_identity(),'backend':'chain-scipy-single-process',
            'approximations':['Finite charge-basis cosine modes','Declared retained local levels and joint excitation projection',
              'Full microwave carriers','Measurement endpoint composition; effective QND freezes local populations'],
            'actual_model':{'mode_labels':model.cell['mode_labels'],'mode_roles':model.cell['mode_roles'],
                'dimensions':list(model.basis.dims),'cap':model.basis.cap,'dimension':model.basis.dimension,
                'encoding':m['encoding'],'propagated_input_labels':r['initial']['labels'],'basis_order':m['basis_order']},
            'duration_ns':result['duration_ns'],'scheduled_duration_ns':result['scheduled_duration_ns'],
            'next_stage':result['next_stage'],'branch_metadata':meta,'summary':summary,'checks':checks,
            'trace':result['trace'],'observations':result['observations'],'energy_reference':result.get('energy_reference'),
            'timing':{'wall_time_s':time.monotonic()-started,'peak_memory_bytes_process_lifetime':int(observed),
                      'memory_observation_scope':'process lifetime; use isolated worker for per-task peak'},
            'checkpoint':'checkpoint.json','resumed_from_checkpoint':resume_from is not None,
            'scientific_qualification':{'calibrated':r.get('calibration') is not None,'numerical_convergence':False,
                'physical_validation':False,'scope':'Completed finite-model calculation; no automatic physical or convergence qualification'},
            'payload_sha256':{name:file_hash(out/name) for name in ('request.json','plan.json','model.json','model.npz','states.npz')}}
    atomic_json(out/'result.json',public);return public


def resume_request(run_directory, output=None):
    run=Path(run_directory).resolve();request=read_json(run/'request.json')
    if output is None:
        i=1
        while (run.parent/(run.name+f'-resume-{i:03d}')).exists(): i+=1
        output=run.parent/(run.name+f'-resume-{i:03d}')
    return run_request(request,output,resume_from=run)


def load_result(path):
    r=inspect_result(path);root=Path(path);root=root if root.is_dir() else root.parent
    if (root/'states.npz').exists():
        with np.load(root/'states.npz',allow_pickle=False) as a: return r,_decode_branches(r['branch_metadata'],a)
    if r.get('backend')=='corrected-pair-cavity':
        from .cavity import load_result as cl
        return cl(root)
    return r,None
