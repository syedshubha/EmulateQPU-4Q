"""Exact-branch programs: pulses, mid-circuit measurement, latency and feedback (0.1.0.dev4).

Follows the declared semantics of digital_twin/adaptive_program.py (the earlier 9q twin), on the chain engine:
- One physical state per reported-record branch, carried as rho_b = V_b W_b V_b^dag and never renormalized.
  Branch probabilities sum to the input trace; any compression discard is recorded.
- One laboratory clock. Every drive carrier is referenced to the absolute program clock (propagate_states
  carrier_time_offset_ns), so splitting a schedule into segments does not change it.
- measure: the full retained Hamiltonian (plus any declared concurrent schedule) evolves for the measurement duration,
  then the instrument acts instantaneously (the 9q-twin convention). Outcomes are classical record entries.
- Conditional operations use reported outcomes only. A branch that does not match still idles for the same
  duration. The program feedback latency is enforced: a conditioned operation starts no earlier than
  latency after the end of the measurement it reads (an idle wait is inserted when needed).
- virtual_z(theta) on qubit q is a software frame update beta_q -= theta (logical Rz(theta) = diag(e^{-i theta/2},
  e^{i theta/2}) in the metrics convention: logical = exp(-i sum beta_q b_q) * physical). Later drives on q are
  played with phase_rad - beta_q.
- observe: records branch probabilities, idle-Hamiltonian term energies and populations at that clock time.

Numerical model only. Replacement reset channels are explicit; no relaxation or classical-electronics model is included.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np

from .model import digest, _real
from .evolve import propagate_states, validate_schedule_v2
from . import measurement as ms
from .energy import EnergyTerms

PROGRAM_SCHEMA = 'emulateqpu.chain.program.v1'
OPS = ('evolve', 'wait', 'measure', 'virtual_z', 'observe', 'reset')
MEASUREMENT_TIME_TOLERANCE_NS = 1e-9


def measurement_interval_ns(operation):
    """Canonical interval, validated without allocating a response or model.

    Effective-QND duration is the response duration (default 300 ns). An
    explicitly supplied operation duration must agree within 1e-9 ns; the
    response duration is always returned. The readout drive must fit that
    window. Ideal projective measurements default to instantaneous.
    """
    inst = operation.get('instrument', {'model': 'ideal_projective'})
    name = inst.get('model', 'ideal_projective')
    if name == 'dispersive_qnd_heterodyne':
        response = _real(inst.get('duration_ns', 300.), 'QND response duration', .1, 2000.)
        _real(inst.get('drive_duration_ns', min(200., response)), 'QND readout drive duration', 0., response)
        supplied = _real(operation.get('duration_ns', response), 'measurement duration', 0., 20000.)
        if abs(supplied-response) > MEASUREMENT_TIME_TOLERANCE_NS:
            raise ValueError('Measurement duration must match the QND instrument response interval within 1e-9 ns')
        T = response
    elif name == 'ideal_projective':
        T = _real(operation.get('duration_ns', 0.), 'measurement duration', 0., 20000.)
    else:
        raise ValueError('Unsupported measurement instrument')
    concurrent = operation.get('concurrent')
    if concurrent is not None:
        ct = _real(concurrent.get('duration_ns'), 'concurrent duration', 0., 20000.)
        if abs(ct-T) > MEASUREMENT_TIME_TOLERANCE_NS:
            raise ValueError('Concurrent schedule must span the measurement window')
    return T


def encoding_labels(model):
    """Column labels, including a deliberately selected computational subset."""
    labels = model.analysis.get('encoding_labels')
    if labels is not None:
        return list(labels)
    return [''.join(map(str, b)) for b in np.ndindex((2,)*len(model.data_modes))]


def initial_branch(model, spec):
    nd = len(model.data_modes); kind = spec.get('kind', 'dressed_computational')
    if kind in ('dressed_computational', 'selected_encoding'):
        if kind == 'dressed_computational' and model.analysis.get('encoding_kind', 'dressed') != 'dressed':
            raise ValueError('A bare selected encoding is not dressed_computational; select selected_encoding explicitly')
        bits = spec.get('bits', '0'*nd)
        if not isinstance(bits, str) or len(bits) != nd or set(bits)-{'0', '1'}: raise ValueError('bits: one 0/1 per data qubit')
        if bits not in encoding_labels(model): raise ValueError('Initial bitstring is absent from the selected encoding')
        psi = model.encoding[:, encoding_labels(model).index(bits)]
    elif kind == 'dressed_superposition':
        selected = encoding_labels(model); c = np.zeros(len(selected), complex)
        for bits, a in spec['amplitudes'].items():
            if bits not in selected: raise ValueError('Initial bitstring is absent from the selected encoding')
            c[selected.index(bits)] = ms._cnum(a, 'amplitude', 1.)
        if abs(np.linalg.norm(c)-1) > 1e-9: raise ValueError('Superposition amplitudes must be normalized')
        psi = model.encoding@c
    elif kind == 'vector':
        psi = np.asarray(spec['real'], float)+1j*np.asarray(spec['imag'], float)
        if psi.shape != (model.basis.dimension,) or abs(np.linalg.norm(psi)-1) > 1e-9: raise ValueError('Normalized retained-basis vector required')
    else: raise ValueError('initial kind must be dressed_computational, selected_encoding, dressed_superposition or vector')
    return ms.pure_branch(psi, nd)


def _condition_ok(cond, record):
    return cond is None or record.get(cond['key']) == cond['equals']


def _frame_schedule(model, schedule, frame):
    s = deepcopy(schedule); labels = model.cell['mode_labels']
    for d in s.get('drives', []):
        i = labels.index(d['mode'])
        if i in model.data_modes: d['phase_rad'] = d.get('phase_rad', 0.)-float(frame[model.data_modes.index(i)])
    s.pop('output_phases_rad', None)
    return s


def _idle(T):
    return dict(duration_ns=T, flux=[], drives=[])


def _evolve_group(model, branches, schedule, t0, solver):
    """Evolve branches that share a schedule and frame with one stacked propagation."""
    if not branches or schedule['duration_ns'] == 0: return dict(nfev=0, seconds=0., norm_error=0.)
    sizes = [b.V.shape[1] for b in branches]
    V = np.hstack([b.V for b in branches])
    if V.shape[1] == 0: return dict(nfev=0, seconds=0., norm_error=0.)
    o = propagate_states(model, schedule, V, carrier_time_offset_ns=t0, **solver)
    k = 0
    for b, r in zip(branches, sizes): b.V = o['final'][:, k:k+r]; k += r
    return dict(nfev=o['nfev'], seconds=o['elapsed_s'], norm_error=o['norm_error'], engine=o['engine'], schedule_sha256=o['schedule_sha256'])


def _evolve_all(model, branches, schedule_for, t0, T, solver):
    """schedule_for(branch) -> schedule (before framing) or None for idle. Groups by (schedule digest, frame)."""
    groups = {}
    for b in branches:
        raw = schedule_for(b)
        s = _idle(T) if raw is None else _frame_schedule(model, raw, b.frame)
        key = (digest(s), tuple(np.round(b.frame, 15)))
        groups.setdefault(key, (s, []))[1].append(b)
    ev = []
    for s, bs in groups.values(): ev.append(_evolve_group(model, bs, s, t0, solver))
    return ev


def compile_program(model, program):
    if not isinstance(program, dict): raise ValueError('Program object required')
    allowed = {'schema', 'initial', 'operations', 'feedback_latency_ns', 'max_branches', 'compress_tol', 'prune_probability', 'regions', 'solver', 'purpose'}
    if set(program)-allowed: raise ValueError('Unknown program fields: '+str(sorted(set(program)-allowed)))
    p = deepcopy(program); p['schema'] = PROGRAM_SCHEMA
    ops = p.get('operations')
    if not isinstance(ops, list) or not 1 <= len(ops) <= 64: raise ValueError('1..64 operations required')
    latency = _real(p.get('feedback_latency_ns', 0.), 'feedback latency', 0., 100000.)
    labels = model.cell['mode_labels']
    keys = {}; stages = []; clock = 0.; bound = 1
    for index, raw in enumerate(ops):
        op = deepcopy(raw); kind = op.get('op'); cond = op.get('condition')
        if kind not in OPS: raise ValueError('op must be one of '+', '.join(OPS))
        if cond is not None:
            if kind in ('measure', 'observe'): raise ValueError(kind+' cannot be conditional')
            if not isinstance(cond, dict) or set(cond) != {'key', 'equals'} or cond['key'] not in keys or cond['equals'] not in keys[cond['key']]['labels']:
                raise ValueError('Condition must name an earlier measurement key and one of its outcome labels')
            ready = keys[cond['key']]['end_ns']+latency
            if clock < ready-1e-12:
                stages.append(dict(index=index, op='latency_wait', duration_ns=ready-clock, start_ns=clock, end_ns=ready, condition=None))
                clock = ready
        st = dict(index=index, op=kind, condition=cond, start_ns=clock)
        if kind == 'evolve':
            if set(op)-{'op', 'schedule', 'condition'}: raise ValueError('evolve takes schedule and condition only')
            s = validate_schedule_v2(model, op['schedule']); st.update(schedule=op['schedule'], duration_ns=s['duration_ns'])
        elif kind == 'wait':
            if set(op)-{'op', 'duration_ns', 'condition'}: raise ValueError('wait takes duration_ns and condition only')
            st.update(duration_ns=_real(op['duration_ns'], 'wait', 0., 100000.))
        elif kind == 'virtual_z':
            if set(op)-{'op', 'qubit', 'angle_rad', 'condition'}: raise ValueError('virtual_z takes qubit, angle_rad, condition')
            q = ms.data_index(model, op['qubit']); st.update(qubit=op['qubit'], data_position=model.data_modes.index(q),
                                                          angle_rad=_real(op['angle_rad'], 'virtual Z angle'), duration_ns=0.)
        elif kind == 'observe':
            if set(op)-{'op', 'label'}: raise ValueError('observe takes label only')
            st.update(label=str(op.get('label', index)), duration_ns=0.)
        elif kind == 'reset':
            if set(op)-{'op', 'model', 'modes', 'state', 'condition', 'max_output_bytes'}:
                raise ValueError('Unknown replacement reset fields')
            reset_model = op.get('model')
            if reset_model not in ('local_replacement', 'fixed_state_replacement'):
                raise ValueError('Reset model must be local_replacement or fixed_state_replacement; physical reset uses conditional evolve')
            reset_modes = op.get('modes')
            if reset_model == 'local_replacement':
                if not isinstance(reset_modes, list) or not reset_modes or len(set(reset_modes)) != len(reset_modes) or any(q not in labels for q in reset_modes):
                    raise ValueError('Local replacement requires unique mode labels')
            elif reset_modes is not None:
                raise ValueError('Fixed state replacement has no modes field')
            max_bytes = op.get('max_output_bytes', 256*1024**2)
            if type(max_bytes) is not int or max_bytes < 1:
                raise ValueError('Reset max_output_bytes must be a positive integer')
            st.update(reset_model=reset_model, modes=reset_modes, state=op.get('state'),
                      max_output_bytes=max_bytes, duration_ns=0.)
        else:
            if set(op)-{'op', 'key', 'qubits', 'instrument', 'duration_ns', 'concurrent'}: raise ValueError('Unknown measure fields')
            key = op.get('key')
            if not isinstance(key, str) or not key or key in keys: raise ValueError('Measurement keys must be unique nonempty strings')
            inst = dict(op.get('instrument', {'model': 'ideal_projective'})); model_name = inst.get('model', 'ideal_projective')
            authoritative_T = measurement_interval_ns(op)
            qubits = op.get('qubits')
            if model_name == 'ideal_projective':
                if set(inst)-{'model', 'basis', 'resolution'}: raise ValueError('Unknown ideal instrument keys')
                outs = ms.ideal_outcomes(model, qubits, inst.get('basis', 'local_levels'), inst.get('resolution', 'computational_plus_leakage'))
                lbl = [o[0] for o in outs]; default_T = 0.
            elif model_name == 'dispersive_qnd_heterodyne':
                outcome = inst.get('outcome', 'classified')
                if outcome not in ('classified', 'unconditional', 'record'):
                    raise ValueError('QND outcome must be classified, unconditional or record')
                seed = inst.get('seed', 0)
                if type(seed) is not int or seed < 0:
                    raise ValueError('QND trajectory seed must be a nonnegative integer')
                cfg = {k: v for k, v in inst.items() if k not in ('outcome', 'classifier', 'record', 'seed')}
                if 'channels' not in cfg: cfg = {**ms.qnd_defaults(qubits), **cfg}
                cfg['duration_ns'] = authoritative_T
                cfg['drive_duration_ns'] = inst.get('drive_duration_ns', min(200., authoritative_T))
                if sorted(ch['reads'] for ch in cfg['channels']) != sorted(qubits): raise ValueError('One QND channel per measured qubit required')
                qi, lab = ms.qnd_instrument(model, cfg)
                lbl = {'classified': [''.join(map(str, b)) for b in np.ndindex((2,)*len(qubits))], 'unconditional': ['unrecorded'], 'record': ['record']}[outcome]
                st.update(qnd=qi, qnd_labels=lab, qnd_outcome=outcome, qnd_classifier=inst.get('classifier'), qnd_record=inst.get('record'),
                          qnd_seed=seed)
                default_T = qi.metadata['duration_ns']
                if [qi.channel_qubit_modes[k] for k in range(len(qubits))] != [ms.data_index(model, q) for q in qubits]:
                    raise ValueError('List channels in the order of qubits')
            else: raise ValueError('instrument model must be ideal_projective or dispersive_qnd_heterodyne')
            T = authoritative_T
            if model_name == 'dispersive_qnd_heterodyne' and abs(T-default_T) > MEASUREMENT_TIME_TOLERANCE_NS:
                raise ValueError('Measurement duration must match the QND instrument response interval within 1e-9 ns')
            conc = op.get('concurrent')
            if conc is not None:
                s = validate_schedule_v2(model, conc)
                if abs(s['duration_ns']-T) > MEASUREMENT_TIME_TOLERANCE_NS: raise ValueError('Concurrent schedule must span the measurement window')
            # A duration accepted within tolerance still uses the authoritative
            # instrument endpoint for control, outcome availability and feedback.
            if model_name == 'dispersive_qnd_heterodyne': T = default_T
            st.update(key=key, qubits=list(qubits), instrument=inst, labels=lbl, duration_ns=T, concurrent=conc)
            if model_name == 'ideal_projective': st['ideal_outcomes'] = outs
            bound *= len(lbl)
        clock += st['duration_ns']; st['end_ns'] = clock; stages.append(st)
        if kind == 'measure': keys[st['key']] = dict(labels=st['labels'], end_ns=clock)
    cap = p.get('max_branches', 64)
    if type(cap) is not int or not 1 <= cap <= 256 or bound > cap: raise ValueError(f'Reported branches ({bound}) exceed max_branches <= 256')
    return p, stages, clock


def run_program(model, program, *, energies=None, process_inputs=None,
                checkpoint_callback=None, resume_state=None, stop_after_stages=None):
    """process_inputs (D x r): process mode. The columns are labelled inputs carried through every branch and Kraus
    block; program['initial'] is then ignored."""
    started = time.monotonic()
    p, stages, total = compile_program(model, program)
    solver = dict(rtol=_real(p.get('solver', {}).get('rtol', 1e-10), 'rtol', 1e-13, 1e-6), atol=_real(p.get('solver', {}).get('atol', 1e-12), 'atol', 1e-15, 1e-8))
    samples = p.get('solver', {}).get('samples', 2)
    if type(samples) is not int or not 2 <= samples <= 20000:
        raise ValueError('Solver samples must be an integer 2..20000')
    solver['samples'] = samples
    input_columns = (process_inputs.shape[1] if process_inputs is not None else
                     (resume_state['branches'][0].block if resume_state is not None and resume_state['branches'] else 1))
    if input_columns > 1 and any(s.get('qnd_outcome') == 'record' for s in stages):
        raise ValueError('A normalized continuous-record trajectory is not a multi-input channel; select one physical input')
    ctol = _real(p.get('compress_tol', 1e-14), 'compress tolerance', 0., 1e-6)
    prune = _real(p.get('prune_probability', 0.), 'prune probability', 0., 1e-9)
    regions = p.get('regions')
    E = energies if energies is not None else EnergyTerms(model)
    program_id = digest(p)
    if stop_after_stages is not None and (type(stop_after_stages) is not int or stop_after_stages < 1):
        raise ValueError('stop_after_stages must be a positive number of additional compiled stages')
    if resume_state is None:
        branches = [initial_branch(model, p.get('initial', {})) if process_inputs is None else ms.process_branch(process_inputs, len(model.data_modes))]
        p_in = branches[0].probability(); trace = []; pruned = 0.; discarded_total = 0.; observations = []; next_stage = 0; clock = 0.
    else:
        if resume_state.get('program_sha256') != program_id or resume_state.get('model_identity') != model.identity():
            raise ValueError('Checkpoint model/program identity mismatch')
        next_stage = resume_state['next_stage']; clock = resume_state['clock_ns']
        if type(next_stage) is not int or not 0 <= next_stage <= len(stages):
            raise ValueError('Invalid checkpoint event cursor')
        expected_clock = 0. if next_stage == 0 else stages[next_stage-1]['end_ns']
        if abs(clock-expected_clock) > 1e-12:
            raise ValueError('Checkpoint laboratory clock disagrees with its event cursor')
        branches = [b.copy() for b in resume_state['branches']]
        p_in = resume_state['input_trace']; trace = deepcopy(resume_state['trace'])
        pruned = resume_state['pruned_probability']; observations = deepcopy(resume_state['observations'])
        discarded_total = resume_state['compression_discarded_trace']
    completed_here = 0
    for cursor in range(next_stage, len(stages)):
        st = stages[cursor]
        t0 = st['start_ns']; T = st['duration_ns']; ev = dict(index=st['index'], op=st['op'], start_ns=t0, end_ns=st['end_ns'])
        if st['op'] in ('wait', 'latency_wait'):
            sel = lambda b: None
            ev['evolution'] = _evolve_all(model, branches, sel, t0, T, solver)
        elif st['op'] == 'evolve':
            sch = st['schedule']; cond = st['condition']
            ev['evolution'] = _evolve_all(model, branches, lambda b: sch if _condition_ok(cond, b.record) else None, t0, T, solver)
            ev['selected_records'] = [b.record for b in branches if _condition_ok(cond, b.record)]
        elif st['op'] == 'virtual_z':
            for b in branches:
                if _condition_ok(st['condition'], b.record): b.frame[st['data_position']] -= st['angle_rad']
        elif st['op'] == 'observe':
            obs = dict(label=st['label'], clock_ns=t0, branches=[])
            for b in branches:
                rep = E.report(b, regions)
                rep.update(record=b.record, populations=_populations(model, b))
                obs['branches'].append(rep)
            observations.append(obs); ev['observation'] = st['label']
        elif st['op'] == 'reset':
            for k, b in enumerate(branches):
                if _condition_ok(st['condition'], b.record):
                    branches[k] = ms.replacement_reset(model, b, modes=st['modes'], state=st['state'],
                        kind='local' if st['reset_model'] == 'local_replacement' else 'full',
                        max_output_bytes=st['max_output_bytes'])
            ev['reset_model'] = st['reset_model']
        else:
            conc = st['concurrent']
            ev['evolution'] = _evolve_all(model, branches, lambda b: conc, t0, T, solver)
            new = []; ev['instrument'] = []
            for b in branches:
                if 'qnd' in st:
                    outs, iev = ms.apply_qnd(b, st['qnd'], st['qnd_labels'], st['qnd_outcome'], classifier_spec=st['qnd_classifier'],
                                             record=st['qnd_record'], seed=st['qnd_seed'])
                else:
                    outs = ms.apply_ideal(b, st['ideal_outcomes']); iev = dict(instrument='ideal_projective')
                masses = {}
                for label, nb in outs:
                    prior_discard = nb.discarded_trace
                    nb = ms.compress(nb, ctol)
                    # A branch's inherited history is shared by its descendants;
                    # only new loss at this event enters the global ledger.
                    discarded_total += nb.discarded_trace-prior_discard
                    nb.record[st['key']] = label
                    pr = nb.probability(); masses[label] = pr
                    if pr <= prune and prune > 0: pruned += pr; continue
                    new.append(nb)
                iev.update(parent_record=b.record, parent_probability=b.probability(), outcome_masses=masses)
                ev['instrument'].append(iev)
            branches = new
        trace.append(ev)
        clock = st['end_ns']; next_stage = cursor+1; completed_here += 1
        checkpoint = dict(schema='emulateqpu.chain.checkpoint.v1', model_identity=model.identity(),
            program_sha256=program_id, next_stage=next_stage, clock_ns=clock,
            branches=[b.copy() for b in branches], input_trace=p_in, trace=deepcopy(trace),
            observations=deepcopy(observations), pruned_probability=pruned,
            compression_discarded_trace=discarded_total,
            solver=solver, program=p, rng_policy='Explicit per-measurement seed in the immutable program')
        if checkpoint_callback is not None: checkpoint_callback(checkpoint)
        if stop_after_stages is not None and completed_here >= stop_after_stages: break
    probs = [b.probability() for b in branches]
    total_p = float(sum(probs)+pruned); discarded = float(discarded_total)
    return dict(schema=PROGRAM_SCHEMA+'.result', program_sha256=program_id, duration_ns=clock,
                scheduled_duration_ns=total, complete=(next_stage == len(stages)), next_stage=next_stage, branches=branches,
                records=[b.record for b in branches], probabilities=probs, probability_total=total_p, input_trace=p_in,
                probability_error=abs(total_p-p_in), pruned_probability=pruned, compression_discarded_trace=discarded,
                probability_balance_error=abs(total_p+discarded-p_in),
                discard_accounting='Summed input-trace units; global event ledger counts shared history once',
                observations=observations, trace=trace, energy_decomposition_error_GHz=E.decomposition_error_GHz,
                energy_reference=getattr(E, 'reference', {'kind': 'caller_supplied_energy_terms'}),
                runtime_s=time.monotonic()-started, stages=[{k: v for k, v in s.items() if k not in ('qnd', 'qnd_labels', 'ideal_outcomes')} for s in stages])


def _populations(model, b):
    """Normalized bare level marginals of every mode and dressed computational populations of the branch."""
    pr = b.probability(); d = b.diagonal()/max(pr, 1e-300); S = model.basis.states; labels = model.cell['mode_labels']
    marg = {labels[i]: np.bincount(S[:, i], weights=d, minlength=model.basis.dims[i]).tolist() for i in range(len(labels))}
    A = model.encoding.conj().T@b.V; comp = np.real(np.einsum('kj,jl,kl->k', A, b.W, A.conj()))/max(pr, 1e-300)
    nd = len(model.data_modes)
    full_dressed = model.analysis.get('encoding_kind', 'dressed') == 'dressed' and model.encoding.shape[1] == 2**nd
    return dict(bare_level_marginals=marg, dressed_computational={label: float(v) for label, v in zip(encoding_labels(model), comp)},
                dressed_leakage=float(1-comp.sum()) if full_dressed else None,
                selected_encoding_complement=float(1-comp.sum()) if not full_dressed else None)


def computational_density(model, branch, t_ns):
    """Logical computational block of a branch at program time t: interaction picture w.r.t. the dressed energies,
    then the branch's software frame. Unnormalized (trace = surviving probability)."""
    if model.analysis.get('encoding_kind', 'dressed') != 'dressed':
        raise ValueError('Idle-energy computational frame requires a dressed encoding')
    bits = np.array([list(map(int, s)) for s in encoding_labels(model)])
    Z = np.exp(-1j*(bits@branch.frame))*np.exp(2j*np.pi*model.encoded_energies*t_ns)
    A = Z[:, None]*(model.encoding.conj().T@branch.V)
    return A@branch.W@A.conj().T
