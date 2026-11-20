"""Conservative resource estimates before basis, operators or states are allocated."""
from __future__ import annotations
import json
import math
from typing import Any

from .contracts import CarrierWork, ExperimentPlan, aggregate_limit_bytes, resource_limits, validate_request
from .io import digest


def basis_dimension(dimensions, cap=None, exempt=()):
    """Exact coefficient sum using integer arithmetic, without enumerating states."""
    if cap is None: return math.prod(dimensions)
    if type(cap) is not int or cap < 0: raise ValueError('Nonnegative integer cap required')
    exempt=set(exempt); cap=min(cap,sum(d-1 for i,d in enumerate(dimensions) if i not in exempt))
    counts=[1]+[0]*cap; factor=1
    for i,d in enumerate(dimensions):
        if i in exempt: factor*=d; continue
        updated=[0]*(cap+1)
        for k,n in enumerate(counts):
            if n:
                for j in range(min(d,cap-k+1)): updated[k+j]+=n
        counts=updated
    return sum(counts)*factor


def _carrier_work(operations: list[dict[str, Any]]) -> CarrierWork:
    """Report DOP853 work floors for one nonempty solve per driven stage."""
    from scipy.integrate import DOP853

    work: CarrierWork = {
        'stages': [],
        'unconditional_rhs_evaluations_floor': 0,
        'conditional_rhs_evaluations_floor_if_all_run': 0,
    }
    for index, operation in enumerate(operations):
        kind = operation['op']
        if kind == 'evolve':
            schedule = operation['schedule']
        elif kind == 'measure':
            schedule = operation.get('concurrent')
        else:
            continue
        if schedule is None or not schedule.get('drives'):
            continue
        duration = float(schedule['duration_ns'])
        carrier = max(float(pulse['frequency_GHz']) for pulse in schedule['drives'])
        # The carrier-resolved chain engine caps steps at 1/(16*f). A floor
        # avoids an off-by-one overestimate when a product is near an integer.
        carrier_steps = duration * carrier * 16
        steps = max(1, math.floor(carrier_steps))
        max_step = 1 / (16 * carrier) if carrier_steps > 1 else duration
        rhs = steps * DOP853.n_stages
        conditional = operation.get('condition') is not None
        work['stages'].append({
            'operation_index': index, 'operation': kind, 'conditional': conditional,
            'duration_ns': duration, 'max_carrier_GHz': carrier, 'max_step_ns': max_step,
            'accepted_steps_floor': steps, 'rhs_evaluations_floor': rhs,
        })
        if conditional:
            work['conditional_rhs_evaluations_floor_if_all_run'] += rhs
        else:
            work['unconditional_rhs_evaluations_floor'] += rhs
    return work


def plan_request(request: dict[str, Any], profile: dict[str, Any] | None = None,
                 workers: int = 1) -> ExperimentPlan:
    return _plan_validated_request(validate_request(request), profile=profile, workers=workers)


def _plan_validated_request(r: dict[str, Any], profile: dict[str, Any] | None = None,
                            workers: int = 1) -> ExperimentPlan:
    if type(workers) is not int or workers < 1:
        raise ValueError('Positive worker count required')
    budget, _ = resource_limits(r['resources'])
    aggregate_limit = aggregate_limit_bytes(profile)
    carrier_work = None
    if r['engine']=='cavity':
        from .cavity import plan_request as cp
        estimates = cp(r['cavity'])
        D = estimates['dimension']
        peak = estimates['estimated_memory_bytes']
        columns = 1
        branches = 1
        dims = estimates['local_dimensions']
        counts = {'data': 2, 'coupler': 1, 'cavity': 1}
    else:
        c=r['device']['cell']; m=r['model']; dims=m['dimensions']; labels=c['mode_labels']
        if math.prod(dims)>2**63-1: raise ValueError('Full-tensor basis codes exceed the supported signed 64-bit representation')
        exempt=[labels.index(q) for q in m.get('cap_exempt',[])]
        if len(exempt)!=len(set(exempt)) or any(c['mode_roles'][i]!='coupler' for i in exempt):
            raise ValueError('Cap exemptions must be unique coupler labels')
        D=basis_dimension(dims,m['excitation_cap'],exempt)
        counts={k:c['mode_roles'].count(k) for k in ('data','coupler')};counts['cavity']=0
        columns=len(r['initial']['labels']); enc_columns=len(m['encoding']['labels'])
        for b in m['encoding']['labels']:
            if m['excitation_cap'] is not None and sum(int(bit) for bit in b)>m['excitation_cap']:
                raise ValueError('Excitation cap excludes declared encoding '+b)
        operations=r['program']['operations'] if 'program' in r else [{'op':'evolve','schedule':r['schedule']}]
        if r.get('measurement') is not None: operations=[*operations,{'op':'measure',**r['measurement']}]
        branches=1; qnd_response_workspace=0
        for op in operations:
            if op.get('op')=='measure':
                inst=op.get('instrument',{})
                nq=len(op.get('qubits',[]))
                if inst.get('model')=='dispersive_qnd_heterodyne':
                    channels=inst.get('channels', [{'reads':q,'chi_MHz':{q:1.}} for q in op['qubits']])
                    affected=set(op['qubits'])
                    for ch in channels:
                        affected.update(ch.get('chi_MHz',{ch['reads']:1.}))
                        affected.update(ch.get('level_shifts_MHz',{}))
                    if affected-set(labels): raise ValueError('Unknown QND response mode label')
                    response_labels=math.prod(dims[labels.index(q)] for q in affected)
                    response_bytes=256*response_labels**2+64*response_labels*len(channels)*inst.get('response_order',32)
                    qnd_response_workspace+=response_bytes
                if inst.get('model','ideal_projective')=='ideal_projective':
                    branches*=math.prod(dims[labels.index(q)] for q in op['qubits']) if inst.get('resolution')=='all_levels' else 2**nq+1
                elif inst.get('outcome','classified')=='classified': branches*=2**nq
        maximum=r['resources'].get('max_branches',64)
        if branches>maximum: raise ValueError('Estimated branches exceed declared max_branches')
        if columns>r['resources'].get('max_columns',100000): raise ValueError('Input columns exceed declared limit')
        ec=c['EC_matrix_GHz']; pairs=sum(dims[i]*dims[j] for i in range(len(dims)) for j in range(i+1,len(dims)) if ec[i][j]!=0)
        nnz_bound=D*(1+pairs+2*sum(dims))
        operator=32*nnz_bound + 16*D*len(dims)
        state=16*D*columns
        encoding=16*D*enc_columns
        from emulateqpu._engine.model import DENSE_ENCODING_LIMIT
        dense_encoding=64*D*D if D<=DENSE_ENCODING_LIMIT and m['encoding']['kind']=='dressed' else 16*D*128
        factors=columns;maximum_factors=factors
        for op in operations:
            if op.get('op')=='measure' and op.get('instrument',{}).get('model')=='dispersive_qnd_heterodyne': factors*=D
            if op.get('op')=='reset':
                reset_kind=op.get('model')
                if reset_kind not in ('local_replacement','fixed_state_replacement'): raise ValueError('Unsupported reset model')
                if reset_kind=='local_replacement':
                    chosen=op.get('modes')
                    if not isinstance(chosen,list) or not chosen or len(set(chosen))!=len(chosen) or set(chosen)-set(labels): raise ValueError('Reset needs unique mode labels')
                    reset_dimension=math.prod(dims[labels.index(q)] for q in chosen)
                else: reset_dimension=D
                replacement=op.get('state');rank=1
                # Both accepted JSON forms can describe a mixed replacement:
                # a direct matrix, or its explicit real/imag representation.
                real=replacement.get('real') if isinstance(replacement,dict) else replacement
                if isinstance(real,list) and real and isinstance(real[0],list): rank=reset_dimension
                factors*=reset_dimension*rank
            maximum_factors=max(maximum_factors,factors)
        factor_bytes=16*D*maximum_factors+16*maximum_factors**2
        branch_state=branches*max(factor_bytes*4,state*4)
        workspace=24*state+dense_encoding+branch_state+qnd_response_workspace
        checkpoint_count=len(operations)+sum(op.get('condition') is not None for op in operations)
        request_json_bytes=len(json.dumps(r,allow_nan=False).encode())
        model_export=operator+encoding+8*D*(len(dims)+3)+24*sum(d*d for d in dims)
        checkpoint_payload=checkpoint_count*branch_state
        # Request/program/source metadata is repeated in each checkpoint. A
        # separate per-stage allowance covers growing traces and record logs;
        # this remains an estimate, not a filesystem quota guarantee.
        metadata_allowance=(checkpoint_count+3)*request_json_bytes+checkpoint_count*65536
        output=16*D*columns*r['solver']['samples']+branch_state+model_export+checkpoint_payload+metadata_allowance
        peak=128*1024**2+operator+encoding+state+workspace+output
        estimates={'state_block_bytes':state,'encoding_bytes':encoding,'operators_bytes':operator,
                   'solver_workspace_bytes':workspace,'qnd_response_workspace_bytes':qnd_response_workspace,'output_bytes':output,
                   'dense_density_bytes':16*D*D,'dense_process_bytes':16*D**4,
                   'model_export_bytes':model_export,'checkpoint_count_upper_bound':checkpoint_count,
                   'checkpoint_payload_bytes':checkpoint_payload,'metadata_output_allowance_bytes':metadata_allowance,
                   'branch_factor_columns_upper_bound':maximum_factors,'operator_nnz_upper_estimate':nnz_bound,'saved_samples':r['solver']['samples']}
        if output>r['resources'].get('max_output_mb',1024)*1024**2: raise ValueError('Output estimate exceeds budget')
        carrier_work = _carrier_work(operations)
    per=int(peak);aggregate=workers*per
    if per>budget: raise ValueError(f'Preallocation rejection: estimated {per} bytes exceeds {budget} byte worker budget')
    if aggregate_limit is not None and aggregate > aggregate_limit:
        raise ValueError('Aggregate worker memory exceeds profile budget')
    return {'schema':'emulateqpu.plan.v1','request_sha256':digest(r),'engine':r['engine'],
            'dimension':int(D),'local_dimensions':dims,'mode_counts':counts,
            'state_columns':columns,'branches_upper_bound':branches,'workers':workers,'numerical_threads':1,
            'estimated_peak_memory_bytes':per,'per_worker_memory_bytes':per,'aggregate_memory_bytes':aggregate,
            'wall_time_s':r['resources']['wall_time_s'],'memory_budget_bytes':budget,'estimates':estimates,
            'carrier_work':carrier_work,
            'qualification':'Conservative implementation estimates; no measured speed or physical accuracy claim'}
