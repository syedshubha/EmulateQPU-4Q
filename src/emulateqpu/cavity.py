"""Supported finite pair/cavity adapter for the corrected physical-flux model.

The cosine/charge kernels and coherent grouped bath are reused. Construction
and orchestration are extracted from the frozen November 15 readout workflow;
there is no campaign controller, implicit data download or receiver instrument.
"""
from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import hashlib
import math
import time

import numpy as np
from scipy.constants import hbar
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, diags
from threadpoolctl import threadpool_limits

from ._resources import WallTimeLimitError, process_peak_rss_bytes
from .io import atomic_json, digest, file_hash, read_json

SCHEMA = "emulateqpu.cavity.v1"
SOURCE_BINDING = {
    "baseline_commit": "3970e1d30e2f296388cd8f3bc93114aac76e3a3c",
    "corrected_model": {
        "path": "research/code/digital_twin/readout_passive_quantum_20261115.py",
        "sha256": "0ae2049bcb86e2cb04fd6924a99b7a4e31c7b0ba2cecd416832af4941e6e830c",
        "reuse": "joint charging correction, physical flux/voltage conventions, dressed operators",
    },
    "grouped_bath": {
        "path": "research/code/digital_twin/port_bath.py",
        "sha256": "56a65f8f8abcbf38d8bf38b45d5fc7d57c125d3923cf56ad406aa64db058b660",
        "reuse": "unchanged installed physical_flux_bath and bath_liouvillian",
    },
    "generator": {
        "path": "research/code/scripts/readout_passive_quantum_pilot_20261115.py",
        "sha256": "ad1510f7308ac22183f90ac777f2d6af8ef12accb3f1893be54142d126cc0ec8",
        "reuse": "prepare physical-current photon-RWA generator; explicit parameters and guards replace campaign globals",
    },
    "library_changes": "Complete small local tensor and all joint eigenstates; configurable local/cavity levels; no family truncation or historical deadlines.",
}


def _real(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be one finite real number")
    if minimum is not None and value < minimum:
        raise ValueError(name + " must be >= " + str(minimum))
    return float(value)


def example_request():
    """Return an assumed compact pair/cavity fixture, without computation."""
    from emulateqpu._engine.fixtures import demo_cell
    return dict(
        schema=SCHEMA, cell=demo_cell(), dimensions=[3, 3, 3],
        cavity=dict(frequency_GHz=6.8, levels=3, Cmodal_fF=500., Cc_fF=2.,
                    R_ohm=25., lambda_charge_GHz=[.002, .001, -.0015]),
        drive=dict(current_peak_A=2e-11, frequency_GHz=6.8,
                   duration_ns=2., phase_rad=0.),
        ringdown_ns=2., initial=dict(attacker_level=0, victim_superposition=True,
                                    photon_number=0),
        solver=dict(rtol=1e-9, atol=1e-11, max_step_ns=.2, samples=9),
        resources=dict(memory_bytes=512*1024**2, wall_time_s=60., numerical_threads=1),
        purpose="Assumed finite pair/cavity software example; uncalibrated and physically unvalidated.",
    )


def validate_request(value):
    """Validate explicit finite-model assumptions, without model allocation."""
    from emulateqpu._engine.model import validate_cell
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Expected " + SCHEMA)
    r = deepcopy(value)
    allowed = {"schema", "cell", "dimensions", "cavity", "drive", "ringdown_ns", "initial", "solver", "resources", "purpose"}
    if set(r)-allowed:
        raise ValueError("Unknown cavity fields: " + str(sorted(set(r)-allowed)))
    r["cell"] = validate_cell(r["cell"])
    if r["cell"]["mode_roles"] != ["data", "coupler", "data"]:
        raise ValueError("Cavity backend supports exactly a data/coupler/data pair reduction")
    dims = r["dimensions"]
    if not isinstance(dims, list) or len(dims) != 3 or any(type(d) is not int or d < 3 or d > 16 for d in dims):
        raise ValueError("Three explicit local dimensions in 3..16 required")
    c = r["cavity"]
    if not isinstance(c, dict) or set(c) != {"frequency_GHz", "levels", "Cmodal_fF", "Cc_fF", "R_ohm", "lambda_charge_GHz"}:
        raise ValueError("Complete cavity parameters required")
    if type(c["levels"]) is not int or not 2 <= c["levels"] <= 64:
        raise ValueError("Cavity levels must be 2..64")
    for k in ("frequency_GHz", "Cmodal_fF", "R_ohm"):
        if _real(c[k], k, 0.) == 0.:
            raise ValueError(k + " must be positive")
    _real(c["Cc_fF"], "coupling capacitance", 0.)
    if not isinstance(c["lambda_charge_GHz"], list) or len(c["lambda_charge_GHz"]) != 3:
        raise ValueError("One signed charge/cavity coupling per circuit mode required")
    for x in c["lambda_charge_GHz"]:
        _real(x, "lambda_charge_GHz")
    d = r["drive"]
    if not isinstance(d, dict) or set(d) != {"current_peak_A", "frequency_GHz", "duration_ns", "phase_rad"}:
        raise ValueError("Explicit physical Norton-current drive parameters required")
    for k in ("current_peak_A", "duration_ns"):
        _real(d[k], k, 0.)
    if _real(d["frequency_GHz"], "drive frequency", 0.) == 0.:
        raise ValueError("Positive drive frequency required")
    _real(d["phase_rad"], "drive phase")
    tail = _real(r["ringdown_ns"], "finite ringdown", 0.)
    if d["duration_ns"] + tail <= 0.:
        raise ValueError("Positive total evolution time required")
    initial = r["initial"]
    if not isinstance(initial, dict) or set(initial)-{"attacker_level", "victim_superposition", "photon_number", "cavity_superposition"}:
        raise ValueError("Unsupported initial-state fields")
    if type(initial.get("attacker_level", 0)) is not int or not 0 <= initial.get("attacker_level", 0) < dims[0]:
        raise ValueError("Attacker level outside retained basis")
    if type(initial.get("victim_superposition", True)) is not bool:
        raise ValueError("victim_superposition must be bool")
    if "cavity_superposition" in initial:
        a = initial["cavity_superposition"]
        if not isinstance(a, list) or len(a) != c["levels"] or any(not isinstance(z, list) or len(z) != 2 for z in a):
            raise ValueError("cavity_superposition needs [real,imag] per photon level")
        for z in a:
            for x in z:
                _real(x, "cavity amplitude")
        if abs(sum(z[0]**2+z[1]**2 for z in a)-1.) > 1e-10:
            raise ValueError("Normalized cavity amplitudes required")
    else:
        n = initial.get("photon_number", 0)
        if type(n) is not int or not 0 <= n < c["levels"]:
            raise ValueError("Photon number outside retained basis")
    s = r["solver"]
    if not isinstance(s, dict) or set(s) != {"rtol", "atol", "max_step_ns", "samples"}:
        raise ValueError("Explicit solver tolerances, step and samples required")
    for k in ("rtol", "atol", "max_step_ns"):
        if _real(s[k], k, 0.) == 0.:
            raise ValueError("Positive solver tolerance/step required")
    if type(s["samples"]) is not int or not 3 <= s["samples"] <= 10001:
        raise ValueError("Saved samples must be 3..10001")
    res = r["resources"]
    if not isinstance(res, dict) or set(res) != {"memory_bytes", "wall_time_s", "numerical_threads"}:
        raise ValueError("Explicit memory, wall-time and thread limits required")
    if type(res["memory_bytes"]) is not int or res["memory_bytes"] <= 0:
        raise ValueError("Positive integer memory budget required")
    if _real(res["wall_time_s"], "wall time", 0.) == 0.:
        raise ValueError("Positive wall time required")
    if res["numerical_threads"] != 1 or type(res["numerical_threads"]) is not int:
        raise ValueError("Supported bounded cavity execution uses one numerical thread")
    digest(r)
    return r


def plan_request(value):
    """Conservative dense/sparse workspace estimate before joint construction."""
    return _plan_validated_request(validate_request(value))


def _plan_validated_request(r):
    nq = math.prod(r["dimensions"])
    d = nq*r["cavity"]["levels"]
    D = d+1
    samples = r["solver"]["samples"]+1
    terms = dict(density_bytes=16*D*D, operators_bytes=16*d*d*12,
                 grouped_liouvillian_upper_estimate_bytes=32*d**3,
                 solver_workspace_bytes=16*D*D*24,
                 output_bytes=2*16*D*D*samples, runtime_allowance_bytes=128*1024**2)
    estimated = sum(terms.values())
    if estimated > r["resources"]["memory_bytes"]:
        raise MemoryError(f"Cavity plan estimates {estimated} bytes, exceeds {r['resources']['memory_bytes']} byte budget")
    return dict(schema="emulateqpu.cavity.plan.v1", backend="corrected-pair-cavity",
                dimension=D, joint_dimension_without_sink=d, data_modes=2,
                coupler_modes=1, cavity_modes=1, local_dimensions=r["dimensions"]+[r["cavity"]["levels"]],
                excitation_cap=None, propagated_inputs=1, matched_arms=2,
                estimated_memory_bytes=estimated, per_worker_memory_bytes=estimated,
                aggregate_memory_bytes=estimated, storage_terms=terms,
                numerical_threads=1, wall_time_s=r["resources"]["wall_time_s"],
                estimates_are_bounds=False, request_sha256=digest(r))


def _bath_functions():
    path = files("emulateqpu._physics").joinpath("port_bath.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_BINDING["grouped_bath"]["sha256"]:
        raise ValueError("Corrected physical-flux bath source binding changed")
    from emulateqpu._physics.port_bath import physical_flux_bath, bath_liouvillian, series_conductance
    return physical_flux_bath, bath_liouvillian, series_conductance


@threadpool_limits.wrap(limits=1)
def build_model(value):
    """Build a full finite pair/cavity model after admitting its resource plan."""
    from emulateqpu._engine.fixtures import build_model as build_chain
    r = validate_request(value)
    plan = _plan_validated_request(r)
    physical_flux_bath, _, series_conductance = _bath_functions()
    c = r["cavity"]
    fr, N = c["frequency_GHz"], c["levels"]
    corrected_cell = deepcopy(r["cell"])
    lam = np.asarray(c["lambda_charge_GHz"])
    # 4 n^T EC n contains Lambda^2/fr exactly: diagonal P n^2 P comes
    # from the charge-basis cosine solve, never from a cropped n squared.
    ec = np.asarray(corrected_cell["EC_matrix_GHz"])+np.outer(lam, lam)/(4*fr)
    corrected_cell["EC_matrix_GHz"] = ec.tolist()
    corrected_cell["provenance"] = {**corrected_cell.get("provenance", {}),
        "cavity_counterterm": "EC_base+lambda*lambda^T/(4*f_r), before charge-basis diagonalization",
        "base_cell_sha256": digest(r["cell"])}
    chain = build_chain(corrected_cell, dims=r["dimensions"], cap=None)
    dims = tuple(r["dimensions"])
    nq = math.prod(dims)
    Eq, Uq = eigh(chain.H.toarray(), check_finite=False)
    Eq -= Eq[0]
    _, assignment = linear_sum_assignment(-abs(Uq)**2)
    phase = np.where(Uq[np.arange(nq), assignment] < 0, -1., 1.)
    Uq[:, assignment] *= phase
    Cbare = sum(lam[i]*chain.charge_ops[i].toarray() for i in range(3))
    Cq = Uq.T@Cbare@Uq
    a = np.diag(np.sqrt(np.arange(1, N)), 1)
    X, P = a+a.T, 1j*(a-a.T)
    H = np.kron(np.diag(Eq), np.eye(N))+np.kron(np.eye(nq), fr*np.diag(np.arange(N)))+np.kron(Cq, X)
    Eall, V = eigh(H, check_finite=False)
    tuples = np.array(list(np.ndindex(dims)), dtype=int)
    rows = np.array([int(assignment[j])*N+n for j in range(nq) for n in range(N)])
    _, selected = linear_sum_assignment(-abs(V[rows, :])**2)
    W = V[:, selected].copy()
    W *= np.where(W[rows, np.arange(len(rows))] < 0, -1., 1.)
    E = Eall[selected]
    d = len(E)
    W3 = W.reshape(nq, N, d)
    def project(op):
        return W.T@np.einsum("nm,imb->inb", op, W3, optimize=True).reshape(d, d)
    Phi = -project(P)
    physical_N = project(np.diag(np.arange(N)))
    annihilation = project(a)
    labels = np.repeat(tuples, N, axis=0)
    photons = np.tile(np.arange(N), nq)
    fd = r["drive"]["frequency_GHz"]
    local_energies = [t[0] for t in chain.local]
    F = np.array([sum(local_energies[k][f[k]] for k in range(3))+n*fd
                  for f in tuples for n in range(N)])+E[0]
    bath = physical_flux_bath(E, labels, photons, Phi, fr_GHz=fr,
        Cmodal_fF=c["Cmodal_fF"], Cc_fF=c["Cc_fF"], R_ohm=c["R_ohm"], frame_GHz=F)
    kappa = float(series_conductance(2*np.pi*fr, c["Cc_fF"], c["R_ohm"])/c["Cmodal_fF"])
    return dict(request=r, plan=plan, N=N, d=d, E=E, F=F, labels=labels,
                n=photons, Phi=Phi, physical_N=physical_N, annihilation=annihilation,
                bath=bath, corrected_EC_GHz=ec, energy_kappa_per_ns=kappa,
                amplitude_decay_per_ns=kappa/2, chain_identity=chain.identity(),
                source_binding=deepcopy(SOURCE_BINDING),
                model_sha256=digest(dict(cell=r["cell"], dimensions=r["dimensions"],
                                         cavity=r["cavity"], frame_frequency_GHz=fd)),
                maximum_hamiltonian_asymmetry=float(abs(H-H.T).max()),
                joint_eigen_residual_GHz=float(abs(H@W-W*E).max()),
                minimum_label_overlap=float(np.min(abs(W[rows, np.arange(d)])**2)))


def _initial(model):
    r = model["request"]
    D, N = model["d"]+1, model["N"]
    psi = np.zeros(D, dtype=complex)
    initial = r["initial"]
    photons = np.zeros(N, dtype=complex)
    if "cavity_superposition" in initial:
        photons[:] = [complex(*z) for z in initial["cavity_superposition"]]
    else:
        photons[initial.get("photon_number", 0)] = 1.
    vs = (0, 1) if initial.get("victim_superposition", True) else (0,)
    dims = tuple(r["dimensions"])
    for v in vs:
        k = np.ravel_multi_index((initial.get("attacker_level", 0), 0, v), dims)
        psi[k*N:(k+1)*N] = photons/np.sqrt(len(vs))
    return np.outer(psi, psi.conj())


def _generator(model):
    """Parameter-only extraction of the frozen corrected prepare generator."""
    _, bath_liouvillian, _ = _bath_functions()
    r, d = model["request"], model["d"]
    D = d+1
    c, drive = r["cavity"], r["drive"]
    Phi0 = np.sqrt(hbar/(2*c["Cmodal_fF"]*1e-15*2*np.pi*c["frequency_GHz"]*1e9))
    amplitude = drive["current_peak_A"]*Phi0/(2*hbar)*1e-9
    L = bath_liouvillian(model["bath"], d, True)
    angles = np.r_[2*np.pi*(model["E"]-model["F"]), 0.]
    L += diags((-1j*(angles[:, None]-angles[None, :])).ravel(), format="csr")
    gap = model["E"][None, :]-model["E"][:, None]
    mask = (gap > 1e-10)&(abs(gap-drive["frequency_GHz"]) <= .05)
    for k in range(d//model["N"]):
        for n in range(1, model["N"]):
            mask[k*model["N"]+n-1, k*model["N"]+n] = True
    ii, jj = np.nonzero(mask&(abs(model["Phi"]) >= 1e-14))
    aa = -amplitude*model["Phi"][ii, jj]*np.exp(1j*drive["phase_rad"])
    ww = 2*np.pi*(model["F"][ii]-model["F"][jj]+drive["frequency_GHz"])
    order = np.lexsort((np.r_[jj, ii], np.r_[ii, jj]))
    base, frequency = np.r_[aa, aa.conj()][order], np.r_[ww, -ww][order]
    H = coo_matrix((base, (np.r_[ii, jj][order], np.r_[jj, ii][order])), shape=(D, D)).tocsr()
    def rhs(t, y, on):
        rho = y[:D*D].reshape(D, D)
        out = L@y[:D*D]
        if on:
            H.data[:] = base*np.exp(1j*frequency*t)
            out += (-1j*(H@rho-(H.T@rho.T).T)).ravel()
        u = np.exp(2j*np.pi*np.r_[model["F"], 0.]*t)
        lab = u.conj()[:, None]*rho*u[None, :]
        photons = float(np.einsum("ij,ji->", model["physical_N"], lab[:d, :d]).real)
        return np.r_[out, photons]
    return rhs


@threadpool_limits.wrap(limits=1)
def evolve_model(model, *, drive_on=True, initial_density=None, deadline=None):
    """Evolve one density; raw trace and a separate sink remain visible."""
    r = model["request"]
    D = model["d"]+1
    rho = _initial(model) if initial_density is None else np.asarray(initial_density, dtype=complex)
    if rho.shape != (D, D) or not np.isfinite(rho).all() or abs(np.trace(rho)-1) > 1e-10 or abs(rho-rho.conj().T).max() > 1e-10 or np.linalg.eigvalsh(rho).min() < -1e-10:
        raise ValueError("A normalized positive retained-plus-sink density is required")
    if deadline is None:
        deadline = time.monotonic()+r["resources"]["wall_time_s"]
    rhs = _generator(model)
    def guard():
        if time.monotonic() >= deadline:
            raise WallTimeLimitError("Cavity wall-time budget exhausted")
        if process_peak_rss_bytes() > r["resources"]["memory_bytes"]:
            raise MemoryError("Cavity process peak RSS exceeds explicit budget")
    def checked(t, y, on):
        guard()
        return rhs(t, y, on)
    ton = float(r["drive"]["duration_ns"])
    total = ton+float(r["ringdown_ns"])
    times = np.unique(np.r_[np.linspace(0, total, r["solver"]["samples"]), ton])
    y = np.r_[rho.ravel(), 0j]
    saved_times, saved, exposure, nfev = [0.], [rho.copy()], [0.], 0
    edges = sorted(set((0., ton, total)))
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        sample = times[(times > lo)&(times <= hi)]
        sol = solve_ivp(lambda t, y: checked(t, y, drive_on and lo < ton), (lo, hi), y,
                        method="DOP853", rtol=r["solver"]["rtol"], atol=r["solver"]["atol"],
                        max_step=r["solver"]["max_step_ns"], t_eval=sample)
        if not sol.success:
            raise RuntimeError(sol.message)
        nfev += int(sol.nfev)
        y = sol.y[:, -1].copy()
        for t, yy in zip(sol.t, sol.y.T):
            saved_times.append(float(t))
            saved.append(yy[:D*D].reshape(D, D).copy())
            exposure.append(float(yy[-1].real))
    densities = np.array(saved)
    physical_N, field, coherence, leakage = [], [], [], []
    family_shape = tuple(r["dimensions"])
    good = (model["labels"][:, 0] < 2)&(model["labels"][:, 1] == 0)&(model["labels"][:, 2] < 2)
    for t, q in zip(saved_times, densities):
        u = np.exp(2j*np.pi*np.r_[model["F"], 0.]*t)
        lab = u.conj()[:, None]*q*u[None, :]
        physical_N.append(float(np.einsum("ij,ji->", model["physical_N"], lab[:-1, :-1]).real))
        field.append(complex(np.einsum("ij,ji->", model["annihilation"], lab[:-1, :-1])))
        C = 0j
        # As in the frozen readout workflow, coherence is defined in the
        # declared local rotating frame; physical photon/field operators
        # above use the restored laboratory state.
        for attacker in range(family_shape[0]):
            for coupler in range(family_shape[1]):
                i = np.ravel_multi_index((attacker, coupler, 0), family_shape)*model["N"]
                j = np.ravel_multi_index((attacker, coupler, 1), family_shape)*model["N"]
                C += np.trace(q[i:i+model["N"], j:j+model["N"]])
        coherence.append(2*C)
        leakage.append(float(np.diag(q).real[:-1][~good].sum()+q[-1, -1].real))
    return dict(times_ns=np.array(saved_times), densities=densities,
                physical_photons=np.array(physical_N), field_amplitude=np.array(field),
                victim_coherence=np.array(coherence), dressed_computational_leakage=np.array(leakage),
                integrated_physical_photons_ns=np.array(exposure), nfev=nfev)


def run_request(value, output, *, deadline=None):
    """Run driven and exactly schedule-matched idle arms in a fresh directory."""
    r = validate_request(value)
    plan = plan_request(r)
    out = Path(output).expanduser().resolve()
    if out.exists():
        raise ValueError("Cavity output must be a fresh directory")
    started = time.monotonic()
    if deadline is None:
        deadline = started+r["resources"]["wall_time_s"]
    if started >= deadline:
        raise WallTimeLimitError("Cavity wall-time budget exhausted before simulation")
    out.mkdir(parents=True)
    atomic_json(out/"request.json", r, sort_keys=False)
    atomic_json(out/"plan.json", plan, sort_keys=False)
    try:
        model = build_model(r)
        driven = evolve_model(model, deadline=deadline)
        idle = evolve_model(model, drive_on=False, deadline=deadline)
        arrays = {}
        for prefix, arm in (("driven", driven), ("idle", idle)):
            for k, v in arm.items():
                if isinstance(v, np.ndarray):
                    arrays[prefix+"_"+k] = v
        for k in ("E", "F", "labels", "n", "Phi", "physical_N", "annihilation", "corrected_EC_GHz"):
            arrays["model_"+k] = model[k]
        np.savez_compressed(out/"arrays.npz", **arrays)
        checks = {}
        for prefix, arm in (("driven", driven), ("idle", idle)):
            q = arm["densities"]
            checks[prefix] = dict(max_trace_error=float(abs(np.trace(q, axis1=1, axis2=2)-1).max()),
                maximum_hermiticity_error=float(abs(q-q.conj().transpose(0, 2, 1)).max()),
                minimum_density_eigenvalue=float(min(np.linalg.eigvalsh((a+a.conj().T)/2).min() for a in q)))
            checks[prefix]["outcome"] = "PASS" if checks[prefix]["max_trace_error"] < 1e-7 and checks[prefix]["maximum_hermiticity_error"] < 1e-7 and checks[prefix]["minimum_density_eigenvalue"] >= -1e-7 else "FAIL"
        result = dict(schema="emulateqpu.result.v1", backend="corrected-pair-cavity",
            execution_status="completed", complete=True,
            package_version="1.0.0", request_sha256=digest(r), model_sha256=model["model_sha256"],
            basis=dict(local_dimensions=r["dimensions"], cavity_levels=model["N"], dimension=model["d"]+1,
                       sink_index=model["d"], order="q0,coupler,q1,photon; globally assigned dressed ladders, then sink"),
            propagated_inputs=1, checks=checks, plan=plan,
            physical_approximations=["finite charge cosine modes", "full small pair/cavity tensor", "photon drive RWA ±50 MHz plus adjacent same-family ladders", "zero-temperature coherent grouped partially secular Markov bath"],
            charging_correction="EC_total=EC_base+outer(lambda,lambda)/(4*f_r), before local quantization; P*n²*P retained",
            drive="Physical Norton current, H_drive=-Phi*I; not a calibrated receiver",
            energy_kappa_per_ns=model["energy_kappa_per_ns"], amplitude_decay_per_ns=model["amplitude_decay_per_ns"],
            matched_idle=True, finite_ringdown_ns=r["ringdown_ns"],
            excess_physical_photon_us=float((driven["integrated_physical_photons_ns"][-1]-idle["integrated_physical_photons_ns"][-1])/1000),
            sink_population=float(driven["densities"][-1, -1, -1].real), discarded_weight=0.,
            dressed_computational_leakage=float(driven["dressed_computational_leakage"][-1]),
            victim_contrast=float(abs(driven["victim_coherence"][-1])),
            induced_victim_phase_rad=float(np.angle(idle["victim_coherence"][-1]/driven["victim_coherence"][-1])) if min(abs(driven["victim_coherence"][-1]), abs(idle["victim_coherence"][-1])) > 1e-8 else None,
            measurement_records=None, record_scope="Unconditional densities and victim tomography observables; no conditional receiver instrument",
            scientific_qualification=dict(numerical_convergence="NOT_RUN", physical_validation="NOT_RUN", calibrated_gate=False),
            timing=dict(wall_time_s=time.monotonic()-started, nfev=driven["nfev"]+idle["nfev"]),
            observed_peak_memory_bytes=process_peak_rss_bytes(), source_binding=SOURCE_BINDING,
            adapter_source_sha256=file_hash(__file__),
            artifacts={name: file_hash(out/name)
                       for name in ("arrays.npz", "request.json", "plan.json")})
        result["status"] = "PASS" if all(row["outcome"] == "PASS" for row in checks.values()) else "FAIL"
        result["payload_sha256"] = dict(result["artifacts"])
        atomic_json(out/"result.json", result, sort_keys=False)
        return result
    except BaseException as exc:
        atomic_json(out/"failure.json", dict(execution_status="failed", complete=False,
            error=type(exc).__name__, message=str(exc), request_sha256=digest(r)), sort_keys=False)
        raise


def load_result(directory):
    """Verify exported result bytes and reload arrays without numerical work."""
    p = Path(directory)
    result = read_json(p/"result.json")
    if result.get("schema") != "emulateqpu.result.v1" or result.get("backend") != "corrected-pair-cavity":
        raise ValueError("Unsupported cavity result")
    for name, expected in result["artifacts"].items():
        if name not in ("arrays.npz", "request.json", "plan.json") or file_hash(p/name) != expected:
            raise ValueError("Cavity result artifact checksum mismatch")
    with np.load(p/"arrays.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    return result, arrays
