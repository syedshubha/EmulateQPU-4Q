"""Read Palace field outputs and gate a separate, fixed-bias modal model."""
from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..chain import linear_chain_cell
from ..io import atomic_json, digest, file_hash, read_json
from .assembly import FLUX_TABLES, magnetostatic_config
from .design import (ASSEMBLY_DESIGN_SCHEMA, ASSEMBLY_PORT_ORDER, DATA_QUBITS, DESIGN_ID, DESIGN_SCHEMA,
                     LEGACY_DESIGN_ID, LEGACY_DESIGN_SCHEMA, LEGACY_PORT_ORDER, MODE_ORDER,
                     PALACE_COMMIT, PALACE_TAG, PORT_ORDER,
                     SHARED_DESIGN_ID, SHARED_PORT_ORDER)

QUALIFICATION_SCHEMA = "emulateqpu.fullwave.qualification.v1"
MODEL_SCHEMA = "emulateqpu.fullwave.model.v1"
OVERLAP_SCHEMA = "emulateqpu.fullwave.mode-overlap.v1"
LOCAL_RUN_SCHEMA = "emulateqpu.fullwave.local-execution.v1"
STAGE_TABLES = {"electrostatic": ("terminal-C.csv",),
                "eigenmode": ("eig.csv", "port-EPR.csv", "port-Q.csv", "domain-E.csv"),
                "driven-train": ("port-S.csv",), "driven-holdout": ("port-S.csv",),
                "magnetostatic": FLUX_TABLES}
TOLERANCE = {"eigen_backward_error": 1e-7, "mesh_frequency_relative": 1e-3,
             "mesh_mode_overlap_min": .90, "energy_relative": .01,
             "junction_energy_fraction_abs": .01, "loss_power_relative": .01, "scattering_passivity_abs": .01,
             "scattering_reciprocity_abs": .01, "heldout_complex_rms": .01}
READOUT_HOLDOUT_METRIC = "readout_through_heldout_complex_rms_max"
ASSEMBLY_LAYOUTS = {"chip-layout.json": "chip_design_sha256", "carrier-layout.json": "carrier_design_sha256"}
ASSEMBLY_DESIGNS = {f"four-data-line-{name}-assembly-v1": name for name in ("dedicated", "shared")}
ASSEMBLY_BRANCH_MODES = tuple(mode for mode in MODE_ORDER for _ in range(1 if mode in DATA_QUBITS else 2))
ASSEMBLY_SHARED_PAIR = ("readout_q0_in", "readout_q3_out")


def _table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = [column.strip() for column in next(reader, ())]
        if not header or any(not column for column in header) or len(set(header)) != len(header):
            raise ValueError(f"Missing or duplicate Palace column in {path}")
        contents = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Incomplete Palace row in {path}")
            contents.append([float(value) for value in row])
    if not contents:
        raise ValueError(f"Empty Palace table {path}")
    return dict(zip(header, np.asarray(contents).T))


def _column(table, name, *, allow_positive_inf=False, allow_negative_inf=False):
    if name not in table:
        raise ValueError(f"Missing Palace column {name}")
    values = table[name]
    valid = np.isfinite(values)
    if allow_positive_inf:
        valid |= np.isposinf(values)
    if allow_negative_inf:
        valid |= np.isneginf(values)
    if not np.all(valid):
        raise ValueError(f"Non-finite Palace values in {name}")
    return values


def _indices(table, name):
    values = _column(table, name)
    if np.any(values < 0) or np.any(values != np.floor(values)) or len(set(values)) != len(values):
        raise ValueError(f"Invalid or duplicate Palace indices in {name}")
    return values.astype(int)


def _same_indices(table, name, expected):
    if not np.array_equal(_indices(table, name), expected):
        raise ValueError(f"Palace rows in {name} differ from the eigenmode inventory")


def _junction_groups(design):
    modes = [row.get("mode", row["label"]) for row in design["junctions"]]
    return [[index for index, name in enumerate(modes) if name == mode] for mode in MODE_ORDER]


def _assembly_inventory(design, shared):
    branches = design["junctions"]
    labels = [branch["label"] for branch in branches]
    if (tuple(branch["mode"] for branch in branches) != ASSEMBLY_BRANCH_MODES
            or [branch["port"] for branch in branches] != list(range(101, 111))
            or len(set(labels)) != len(labels)
            or len(design["island_order"]) != 14 or len(set(design["island_order"])) != 14):
        raise ValueError("Assembly requires fourteen islands and ten ordered junction/SQUID branches")
    for branch in branches:
        numbers = [branch[key] for key in ("EJ_GHz", "inductance_H", "capacitance_F", "bias_Phi0")]
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in numbers)
                or min(numbers[:3]) <= 0 or not 0 <= numbers[-1] < .45):
            raise ValueError(f"Invalid assembled branch parameters for {branch['label']}")
    for group in _junction_groups(design):
        if len({branches[index]["bias_Phi0"] for index in group}) != 1:
            raise ValueError("SQUID branches require one common mode-level idle flux")
    active = set(ASSEMBLY_PORT_ORDER)
    if shared:
        active.difference_update(set(PORT_ORDER[4:]) - set(ASSEMBLY_SHARED_PAIR))
    if any(port.get("active") is not (port["label"] in active) for port in design["ports"]):
        raise ValueError("Assembly port activity differs from the die wire-bond topology")


def _case(directory, revision):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "field-inputs.json")
    design = read_json(directory / "design.json")
    identity = (design.get("schema"), design.get("design_id"))
    assembly = identity[0] == ASSEMBLY_DESIGN_SCHEMA
    topology = ASSEMBLY_DESIGNS.get(identity[1]) if assembly else None
    shared = topology == "shared" if assembly else identity[1] == SHARED_DESIGN_ID
    port_order = {(DESIGN_SCHEMA, DESIGN_ID): PORT_ORDER,
                  (DESIGN_SCHEMA, SHARED_DESIGN_ID): SHARED_PORT_ORDER,
                  (LEGACY_DESIGN_SCHEMA, LEGACY_DESIGN_ID): LEGACY_PORT_ORDER}.get(identity)
    if assembly and topology is not None and design.get("readout_topology") == topology:
        port_order = ASSEMBLY_PORT_ORDER
    if (manifest.get("schema") != "emulateqpu.fullwave.inputs.v1" or manifest.get("refinement") != revision
            or manifest.get("palace") != {"tag": PALACE_TAG, "commit": PALACE_COMMIT}
            or manifest.get("design_sha256") != digest(design)
            or port_order is None or design.get("mode_order") != list(MODE_ORDER)
            or [(port["label"], port["port"]) for port in design["ports"]]
               != [(label, number) for number, label in enumerate(port_order, 1)]):
        raise ValueError("Four-data Palace input identity or ordering differs")
    if assembly:
        _assembly_inventory(design, shared)
    elif [port["label"] for port in design["junctions"]] != list(MODE_ORDER):
        raise ValueError("Four-data Palace junction ordering differs")
    if design["schema"] in (DESIGN_SCHEMA, ASSEMBLY_DESIGN_SCHEMA):
        channels = design["readout_channels"]
        if shared:
            pair = ASSEMBLY_SHARED_PAIR if assembly else ("readout_in", "readout_out")
            expected_channels = [(qubit, *pair) for qubit in DATA_QUBITS]
        else:
            expected_channels = [(qubit, f"readout_{qubit}_in", f"readout_{qubit}_out") for qubit in DATA_QUBITS]
        if [(row["qubit"], row["input_port"], row["output_port"]) for row in channels] != expected_channels:
            raise ValueError("Readout channel order or port mapping differs")
        resonators = {row["resonator"] for row in channels}
        feedlines = {row["feedline"] for row in channels}
        if (len(resonators) != len(channels) or resonators & feedlines
                or not (resonators | feedlines).issubset(design["conductors"])):
            raise ValueError("Readout channels require distinct resonator conductors in the design")
        if (shared and feedlines != {"readout_feedline"}) or (not shared and len(feedlines) != len(channels)):
            raise ValueError("Readout feedline mapping differs")
        if shared and [row["resonator"] for row in channels] != [f"readout_{qubit}_resonator" for qubit in DATA_QUBITS]:
            raise ValueError("Shared readout resonator mapping differs")
    expected = {"design.json", "mesh.msh", "electrostatic.json", "eigenmode.json", "driven-train.json", "driven-holdout.json"}
    if assembly:
        expected.update(ASSEMBLY_LAYOUTS)
    if "flux_transfer" in design:
        if not assembly:
            raise ValueError("Flux transfer requires an assembled die and carrier")
        expected.add("magnetostatic.json")
    if set(manifest["files_sha256"]) != expected:
        raise ValueError("Incomplete frozen Palace input inventory")
    for file, expected_hash in manifest["files_sha256"].items():
        path = directory / file
        if path.is_symlink() or file_hash(path) != expected_hash:
            raise ValueError(f"Palace input integrity mismatch for {file}")
    if assembly and any(digest(read_json(directory / file)) != design.get(key)
                        for file, key in ASSEMBLY_LAYOUTS.items()):
        raise ValueError("Package assembly source layout hashes differ")
    if "flux_transfer" in design:
        _verify_flux_configuration(directory, design, manifest)
    return directory, manifest, design


def _verify_flux_configuration(directory, design, manifest):
    transfer = design["flux_transfer"]
    expected_sources = [
        {"label": row["label"], "index": i, "elements": row["elements"]}
        for i, row in enumerate((row for row in design["ports"] if row["kind"] == "flux"), 1)
    ]
    couplers = [mode for mode in MODE_ORDER if mode not in DATA_QUBITS]
    pickups = [(row["label"], row["surface"], row["index"], row["direction"])
               for row in transfer["pickups"]]
    if (transfer["sources"] != expected_sources
            or [row["label"] for row in expected_sources] != [f"flux_{mode}" for mode in couplers]
            or pickups != [(mode, f"flux_pickup_{mode}", i, "+Z") for i, mode in enumerate(couplers, 1)]
            or any(surface in design["conductors"] or surface not in design["sheets_wkt"]
                   for _, surface, _, _ in pickups)
            or manifest.get("expected_outputs", {}).get("magnetostatic") != list(FLUX_TABLES)):
        raise ValueError("Flux source or passive pickup inventory differs")
    expected = magnetostatic_config(design, manifest["physical_surface_ids"], read_json(directory / "electrostatic.json"))
    if read_json(directory / "magnetostatic.json") != expected:
        raise ValueError("Frozen magnetostatic inputs differ from the flux transfer geometry")


def _stage_output(case, manifest, stage, *, synthetic=False):
    native = case / f"{stage}-output"
    batch = case / f"{stage}-slurm"
    expected_cells = manifest["tetrahedra"]
    assembly = set(ASSEMBLY_LAYOUTS) <= manifest["files_sha256"].keys()
    if batch.exists():
        from ..field_slurm import PALACE_LOG_DIRECTORY, collect

        if native.exists():
            raise ValueError(f"Both local and Slurm outputs exist for {stage}")
        result = collect(batch)
        plan = read_json(batch / "plan.json")
        hashes = manifest["files_sha256"]
        if (not result["complete"] or plan["config"]["sha256"] != hashes[f"{stage}.json"]
                or not any(entry["path"] == plan["mesh"] and entry["sha256"] == hashes["mesh.msh"]
                           for entry in plan["inputs"])):
            raise ValueError(f"Incomplete or mismatched Slurm outputs for {stage}")
        stages = Path(result["raw_directory"])
        if stage == "magnetostatic":
            _field_provenance(stages, FLUX_TABLES, solved_mesh=False, synthetic=False)
            if "logs_sha256" not in result["execution"]:
                raise ValueError("Magnetostatic Slurm outputs require captured Palace logs")
        if "logs_sha256" in result["execution"]:
            preprocessing = _mesh_preprocessing(stages, expected_cells, logs_directory=batch / PALACE_LOG_DIRECTORY)
            if preprocessing["log_sha256"] != result["execution"]["logs_sha256"]:
                raise ValueError(f"Palace mesh preprocessing logs differ for {stage}")
            expected_cells += preprocessing["added_tetrahedra"]
    else:
        stages = native
        if not synthetic:
            receipt = read_json(stages / "fullwave-execution.json")
            hashes = manifest["files_sha256"]
            execution_path = stages / "execution.json"
            execution = read_json(execution_path)
            evidence = _field_provenance(stages, STAGE_TABLES[stage], solved_mesh=stage == "eigenmode",
                                         synthetic=False, expected_mpi_ranks=execution.get("plan", {}).get("mpi_ranks"))
            if (receipt.get("schema") != LOCAL_RUN_SCHEMA or receipt.get("stage") != stage
                    or receipt.get("config_sha256") != hashes[f"{stage}.json"]
                    or receipt.get("mesh_sha256") != hashes["mesh.msh"]
                    or receipt.get("execution_sha256") != file_hash(execution_path)
                    or receipt.get("field_sha256") != evidence
                    or execution.get("status") != "COMPLETE"):
                raise ValueError(f"Local output is stale, incomplete or lacks a bound launch for {stage}")
            if assembly or "mesh_preprocessing" in receipt:
                preprocessing = _mesh_preprocessing(stages, expected_cells)
                if receipt.get("mesh_preprocessing") != preprocessing:
                    raise ValueError(f"Palace mesh preprocessing logs differ for {stage}")
                expected_cells += preprocessing["added_tetrahedra"]
    version = read_json(stages / "palace.json")
    if not synthetic and version.get("ElapsedTime", {}).get("Counts", {}).get("Adaptation") != 0:
        raise ValueError(f"Palace output lacks a zero solution-adaptivity count for {stage}")
    if version.get("Problem", {}).get("MeshElements") != expected_cells:
        raise ValueError(f"Palace solved mesh cell count differs from the input manifest for {stage}")
    return stages


def _mesh_preprocessing(directory, input_cells, *, logs_directory=None):
    """Bind recorded geometric crack repair to the native mesh count."""
    from ..field_slurm import _palace_log_hashes
    from .mesh_study import _native_preprocessing

    logs_directory = directory if logs_directory is None else logs_directory
    log_hashes = _palace_log_hashes(logs_directory)
    preprocessing = _native_preprocessing(logs_directory)
    native = read_json(directory / "palace.json")
    if (native.get("ElapsedTime", {}).get("Counts", {}).get("Adaptation") != 0
            or native["Problem"]["MeshElements"] != input_cells + preprocessing["added_tetrahedra"]):
        raise ValueError("Palace mesh differs from its input and verified geometric preprocessing")
    return {**preprocessing, "log_sha256": log_hashes}


def solve_local(case, stage, *, executable, mpi_ranks=1, launcher=None, build_manifest=None,
                memory_mb=4096, wall_time_s=600):
    """Run a local Palace stage from frozen inputs and register its output evidence."""
    from ..palace import launch_palace, plan_palace

    if stage not in STAGE_TABLES:
        raise ValueError("Unknown full-wave stage")
    revision = read_json(Path(case) / "field-inputs.json")["refinement"]
    case, manifest, _ = _case(case, revision)
    hashes = manifest["files_sha256"]
    config = case / f"{stage}.json"
    if config.name not in hashes:
        raise ValueError(f"Stage {stage} is absent from the frozen case")
    output = case / f"{stage}-output"
    if (case / f"{stage}-slurm").exists():
        raise ValueError(f"A Slurm run already exists for {stage}; use a fresh case")
    plan = plan_palace(config, executable=executable, mpi_ranks=mpi_ranks, launcher=launcher,
                       build_manifest=build_manifest, memory_mb=memory_mb, wall_time_s=wall_time_s)
    if plan["config_sha256"] != hashes[config.name]:
        raise ValueError("Palace configuration changed before launch")
    execution = launch_palace(plan, output)
    if execution["status"] != "COMPLETE":
        return execution
    if file_hash(case / "mesh.msh") != hashes["mesh.msh"]:
        raise ValueError("Palace mesh changed during launch")
    evidence = _field_provenance(output, STAGE_TABLES[stage], solved_mesh=stage == "eigenmode",
                                 synthetic=False, expected_mpi_ranks=plan["mpi_ranks"])
    receipt = {"schema": LOCAL_RUN_SCHEMA, "stage": stage, "status": "COMPLETE",
               "config_sha256": hashes[config.name], "mesh_sha256": hashes["mesh.msh"],
               "execution_sha256": file_hash(output / "execution.json"), "field_sha256": evidence,
               "mesh_preprocessing": _mesh_preprocessing(output, manifest["tetrahedra"])}
    atomic_json(output / "fullwave-execution.json", receipt)
    return receipt


def _field_provenance(directory, files, *, solved_mesh, synthetic, expected_mpi_ranks=None):
    version = read_json(directory / "palace.json")
    expected_tags = {PALACE_TAG, PALACE_TAG.removeprefix("v"), PALACE_COMMIT, PALACE_COMMIT[:7]}
    info = version.get("Problem", {})
    if (version.get("GitTag") not in expected_tags or type(info.get("MPISize")) is not int
            or info["MPISize"] < 1 or type(info.get("MeshElements")) is not int or info["MeshElements"] < 1):
        raise ValueError(f"Missing pinned Palace build and run metadata in {directory}")
    if expected_mpi_ranks is not None and (type(expected_mpi_ranks) is not int or info["MPISize"] != expected_mpi_ranks):
        raise ValueError(f"Palace reported MPI size {info['MPISize']}, expected {expected_mpi_ranks}")
    paths = [directory / name for name in ("palace.json", *files)]
    if solved_mesh:
        meshes = sorted((directory / "gridfunction" / "eigenmode").glob("mesh.*"))
        if len(meshes) != info["MPISize"]:
            raise ValueError("Solved mesh partition count differs from Palace MPI size")
        paths.extend(meshes)
        if not synthetic:
            mode_ids = _indices(_table(directory / "eig.csv"), "m")
            fields = [directory / "gridfunction" / "eigenmode" / f"E_{component}_{mode:06d}.gf.{rank:06d}"
                      for mode in mode_ids for rank in range(info["MPISize"]) for component in ("real", "imag")]
            paths.extend(fields)
    if any(not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve())
           or not path.stat().st_size for path in paths):
        raise ValueError(f"Incomplete or unsafe Palace outputs in {directory}")
    return {path.relative_to(directory).as_posix(): file_hash(path) for path in paths}


def _eigen(directory, design, *, synthetic):
    files = _field_provenance(directory, ("eig.csv", "port-EPR.csv", "port-Q.csv", "domain-E.csv"),
                              solved_mesh=True, synthetic=synthetic)
    eigen, epr, losses, energy = (_table(directory / name) for name in ("eig.csv", "port-EPR.csv", "port-Q.csv", "domain-E.csv"))
    modes = _indices(eigen, "m")
    for table in (epr, losses, energy):
        _same_indices(table, "m", modes)
    f, imag, backward = (_column(eigen, name) for name in ("Re{f} (GHz)", "Im{f} (GHz)", "Error (Bkwd.)"))
    q = _column(eigen, "Q", allow_positive_inf=True)
    participation = np.column_stack([_column(epr, f"p[{junction['port']}]") for junction in design["junctions"]])
    kappa = np.column_stack([_column(losses, f"κ_ext[{port['port']}] (GHz)") for port in design["ports"]])
    if np.any(f <= 0) or np.any(imag < 0) or np.any(backward < 0) or np.any(q <= 0) or np.any(np.abs(participation) > 1.01):
        raise ValueError("Invalid frequencies, quality factors, errors, or junction participation")
    electric, magnetic, capacitance, inductance = (_column(energy, key) for key in
                                                   ("E_elec (J)", "E_mag (J)", "E_cap (J)", "E_ind (J)"))
    electric_total = electric + capacitance
    stored = electric_total + magnetic + inductance
    if np.any(electric_total <= 0) or np.any(stored <= 0) or any(np.any(x < 0) for x in
                                                                  (electric, magnetic, capacitance, inductance)):
        raise ValueError("Invalid electromagnetic energy record")
    expected_q_inverse = 2 * imag / np.hypot(f, imag)
    expected_loss = 2 * imag
    return {"modes": modes, "frequency_GHz": f, "Q": q, "participation": participation,
            "kappa_signed_GHz": kappa, "file_sha256": files,
            "metrics": {"eigen_backward_error": float(backward.max()),
                        "energy_relative": float(np.max(np.abs(2 * electric_total / stored - 1))),
                        "junction_energy_fraction_abs": float(np.max(np.abs(np.abs(participation).sum(axis=1)
                                                                           - inductance / electric_total))),
                        "loss_power_relative": float(np.max(np.abs(np.abs(kappa).sum(axis=1) - expected_loss)
                                                            / np.maximum(expected_loss, 1e-12))),
                        "Q_consistency": bool(np.allclose(1 / q, expected_q_inverse, rtol=1e-5, atol=1e-12))}}


def _scattering(directory, design):
    files = _field_provenance(directory, ("port-S.csv",), solved_mesh=False, synthetic=False)
    table = _table(directory / "port-S.csv")
    frequencies = _column(table, "f (GHz)")
    if len(frequencies) < 2 or np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0):
        raise ValueError("Expected an increasing positive scattering frequency grid")
    ports = [port["port"] for port in design["ports"]]
    response = np.empty((len(frequencies), len(ports), len(ports)), complex)
    for row, destination in enumerate(ports):
        for col, source in enumerate(ports):
            name = f"S[{destination}][{source}]"
            magnitude = _column(table, f"|{name}| (dB)", allow_negative_inf=True)
            phase_deg = _column(table, f"arg({name}) (deg.)")
            if np.any(magnitude > 60):
                raise ValueError("Scattering magnitude exceeds the admissible numeric range")
            response[:, row, col] = 10**(magnitude / 20) * np.exp(1j * np.deg2rad(phase_deg))
    return frequencies, response, files


def _scattering_checks(train, train_response, holdout, holdout_response):
    if np.intersect1d(train, holdout).size or not (train[0] < holdout[0] < holdout[-1] < train[-1]):
        raise ValueError("Held-out frequencies must be disjoint and strictly inside the training band")
    i = np.searchsorted(train, holdout)
    u = ((holdout - train[i - 1]) / (train[i] - train[i - 1]))[:, None, None]
    predicted = (1 - u) * train_response[i - 1] + u * train_response[i]
    squared_error = np.abs(predicted - holdout_response)**2
    full_response = np.concatenate((train_response, holdout_response))
    metrics = {"scattering_passivity_abs": float(max(0, np.linalg.svd(full_response, compute_uv=False).max() - 1)),
               "scattering_reciprocity_abs": float(np.max(np.abs(full_response - full_response.transpose(0, 2, 1)))),
               "heldout_complex_rms": float(np.sqrt(squared_error.mean()))}
    return metrics, np.sqrt(squared_error.mean(axis=0))


def _readout_through_errors(design, port_rms):
    port_index = {port["label"]: index for index, port in enumerate(design["ports"])}
    physical_pairs = {}
    for row in design["readout_channels"]:
        pair = (row["input_port"], row["output_port"])
        physical_pairs.setdefault(pair, []).append(row["qubit"])
    result = []
    for (input_port, output_port), qubits in physical_pairs.items():
        source = port_index[input_port]
        destination = port_index[output_port]
        names = {"qubit": qubits[0]} if len(qubits) == 1 else {"qubits": qubits}
        result.append({**names, "input_port": input_port, "output_port": output_port,
                       "forward_heldout_complex_rms": float(port_rms[destination, source]),
                       "reverse_heldout_complex_rms": float(port_rms[source, destination])})
    return result


def _match_modes(overlap, coarse, fine):
    if (overlap.get("schema") != OVERLAP_SCHEMA
            or overlap.get("coarse_field_sha256") != coarse["file_sha256"]
            or overlap.get("fine_field_sha256") != fine["file_sha256"]
            or overlap.get("coarse_modes") != coarse["modes"].tolist()
            or overlap.get("fine_modes") != fine["modes"].tolist()):
        raise ValueError("Cross-mesh overlap report does not bind the retained solved fields")
    values = np.asarray(overlap["overlap_abs"], dtype=float)
    shape = (len(coarse["modes"]), len(fine["modes"]))
    if shape[0] != shape[1] or values.shape != shape or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid cross-mesh overlap matrix")
    row, col = linear_sum_assignment(values, maximize=True)
    before, after = coarse["frequency_GHz"][row], fine["frequency_GHz"][col]
    return {"mesh_mode_overlap_min": float(values[row, col].min()),
            "mesh_frequency_relative": float(np.max(np.abs(after - before) / after)),
            "mode_pairs": [[int(coarse["modes"][i]), int(fine["modes"][j])] for i, j in zip(row, col)]}


def write_mode_overlap(coarse_dir, fine_dir, samples_path, output):
    """Compare E vectors sampled on common points by a native field evaluator."""
    coarse_path, coarse_manifest, design = _case(coarse_dir, "coarse")
    fine_path, fine_manifest, fine_design = _case(fine_dir, "fine")
    if digest(design) != digest(fine_design):
        raise ValueError("Mode overlap requires the same physical design")
    coarse = _eigen(_stage_output(coarse_path, coarse_manifest, "eigenmode"), design, synthetic=False)
    fine = _eigen(_stage_output(fine_path, fine_manifest, "eigenmode"), design, synthetic=False)
    with np.load(samples_path, allow_pickle=False) as samples:
        points, weights = samples["points_mm"], samples["weights"]
        coarse_fields, fine_fields = samples["coarse_E"], samples["fine_E"]
        if not (np.array_equal(samples["coarse_modes"], coarse["modes"])
                and np.array_equal(samples["fine_modes"], fine["modes"])):
            raise ValueError("Sample mode order differs from Palace mode IDs")
        source_digest = {key: str(samples[key].item()) for key in ("coarse_fields_sha256", "fine_fields_sha256")}
    if source_digest != {"coarse_fields_sha256": digest(coarse["file_sha256"]),
                         "fine_fields_sha256": digest(fine["file_sha256"])}:
        raise ValueError("Sample provenance does not match the retained Palace eigenmode files")
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or weights.shape != (len(points),)
            or coarse_fields.shape != (len(coarse["modes"]), len(points), 3)
            or fine_fields.shape != (len(fine["modes"]), len(points), 3)
            or np.iscomplexobj(points) or np.iscomplexobj(weights)
            or any(not np.isfinite(v).all() for v in (points, weights, coarse_fields, fine_fields))
            or np.any(weights <= 0)):
        raise ValueError("Invalid common-point field sample arrays")
    fields = []
    for vectors in (coarse_fields, fine_fields):
        weighted = (vectors * np.sqrt(weights)[None, :, None]).reshape(len(vectors), -1)
        norms = np.linalg.norm(weighted, axis=1)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ValueError("Cannot normalize a zero sampled electric field")
        fields.append(weighted / norms[:, None])
    matrix = np.minimum(1, np.abs(fields[0].conj() @ fields[1].T))
    result = {"schema": OVERLAP_SCHEMA, "coarse_field_sha256": coarse["file_sha256"],
              "fine_field_sha256": fine["file_sha256"], "coarse_modes": coarse["modes"].tolist(),
              "fine_modes": fine["modes"].tolist(), "overlap_abs": matrix.tolist(),
              "source_samples_sha256": file_hash(samples_path), "sample_count": len(points)}
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh overlap report")
    atomic_json(output, result)
    return result


def _normalized_native_gram(raw, coarse_count, fine_count):
    expected = coarse_count + fine_count
    gram_parts = np.asarray(raw["Gram_complex_re_im"], dtype=float)
    if (raw["count_A"] != coarse_count or raw["count_B"] != fine_count
            or gram_parts.shape != (expected, expected, 2) or not np.isfinite(gram_parts).all()
            or not 0 <= raw["unlocated_volume_fraction"] <= 1e-10
            or not 0 <= raw["material_mismatch_volume_fraction"] <= 1e-10
            or raw["positive_weight_points"] < 1 or not raw["domain_volume_m3"] > 0):
        raise ValueError("Native mode overlap has incomplete mesh coverage or invalid Gram data")
    gram = gram_parts[..., 0] + 1j * gram_parts[..., 1]
    norms = gram.diagonal().real
    if np.any(norms <= 0):
        raise ValueError("Native mode overlap has a zero field norm")
    normalized = gram / np.sqrt(norms[:, None] * norms)
    if (not np.allclose(normalized, normalized.conj().T, atol=1e-10, rtol=0)
            or np.any(np.abs(normalized) > 1 + 1e-10)):
        raise ValueError("Native mode overlap violates Gram symmetry or normalization")
    return normalized


def write_native_overlap(coarse_dir, fine_dir, sampler, output):
    """Run the archived native MFEM integrator on single-rank saved field outputs."""
    coarse_path, coarse_manifest, design = _case(coarse_dir, "coarse")
    fine_path, fine_manifest, fine_design = _case(fine_dir, "fine")
    if digest(design) != digest(fine_design):
        raise ValueError("Native mode overlap requires the same physical design")
    cases = ((coarse_path, coarse_manifest), (fine_path, fine_manifest))
    if any(manifest["material_ids"] != {"substrate": 514, "air": 501} for _, manifest in cases):
        raise ValueError("Material attributes differ from the archived native integrator convention")
    paths = [_stage_output(case, manifest, "eigenmode") for case, manifest in cases]
    eigen = [_eigen(path, design, synthetic=False) for path in paths]
    counts = [len(part["modes"]) for part in eigen]
    for stage, part in zip(paths, eigen):
        if (read_json(stage / "palace.json")["Problem"]["MPISize"] != 1
                or not np.array_equal(part["modes"], np.arange(1, len(part["modes"]) + 1))):
            raise ValueError("The archived overlap helper requires single-rank saves and consecutive modes starting at 1")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh overlap directory")
    executable = Path(sampler).expanduser().resolve(strict=True)
    executable_sha = file_hash(executable)
    with TemporaryDirectory(prefix="emulateqpu-mode-overlap-") as temporary:
        benchmark = subprocess.run([str(executable), "--benchmark"], check=True, timeout=60, capture_output=True)
        benchmark_path = Path(temporary) / "benchmark.json"
        benchmark_path.write_bytes(benchmark.stdout)
        known = read_json(benchmark_path)
        for key in ("q3", "q4"):
            normalized = _normalized_native_gram(known[key], 2, 2)
            expected_cross = known["expected_cross"] / np.sqrt(np.prod(known["expected_norms"]))
            if (not np.allclose(normalized[:2, 2:], normalized[:2, :2], atol=1e-10, rtol=0)
                    or abs(normalized[0, 1] - expected_cross) > 1e-10):
                raise ValueError("Archived native overlap helper failed its analytic fixture")
        matrices, records = [], []
        for rule in (3, 4):
            raw = Path(temporary) / f"duffy{rule}.json"
            arguments = [str(executable)]
            for path, count in zip(paths, counts):
                arguments.extend((str(path / "gridfunction" / "eigenmode"), str(count)))
            subprocess.run([*arguments, str(rule), str(raw)], check=True, timeout=300, capture_output=True)
            record = read_json(raw)
            matrices.append(_normalized_native_gram(record, *counts))
            records.append(record)
        delta = float(np.max(np.abs(matrices[0] - matrices[1])))
        if delta > .001:
            raise ValueError(f"Cross-mesh field overlap changed by {delta:.4g} between Duffy3 and Duffy4")
        if file_hash(executable) != executable_sha or any(
            _eigen(path, design, synthetic=False)["file_sha256"] != original["file_sha256"]
            for path, original in zip(paths, eigen)
        ):
            raise ValueError("Overlap inputs changed while integrating")
    count = counts[0]
    report = {"schema": OVERLAP_SCHEMA, "coarse_field_sha256": eigen[0]["file_sha256"],
              "fine_field_sha256": eigen[1]["file_sha256"], "coarse_modes": eigen[0]["modes"].tolist(),
              "fine_modes": eigen[1]["modes"].tolist(),
              "overlap_abs": np.minimum(1, np.abs(matrices[1][:count, count:])).tolist(),
              "native_helper_sha256": executable_sha, "duffy_normalized_overlap_difference": delta,
              "samples": [record["positive_weight_points"] for record in records]}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "benchmark.json", known)
    for rule, record in zip((3, 4), records):
        atomic_json(output / f"duffy{rule}.json", record)
    atomic_json(output / "overlap.json", report)
    return report


def bind_electrostatic(directory, design, *, synthetic=False):
    """Load the grounded terminated electrostatic comparison at the same idle junction biases."""
    from emulateqpu._physics.chain_family import reduce_islands

    files = _field_provenance(directory, ("terminal-C.csv",), solved_mesh=False, synthetic=False)
    count = len(design["island_order"])
    table = _table(directory / "terminal-C.csv")
    if not np.array_equal(_indices(table, "i"), np.arange(1, count + 1)):
        raise ValueError("Capacitance terminal inventory differs from the design")
    capacitance = np.column_stack([_column(table, f"C[i][{i}] (F)") for i in range(1, count + 1)])
    if np.max(capacitance - np.diag(np.diag(capacitance))) > 1e-25:
        raise ValueError("Electrostatic mutual capacitances have the wrong sign")
    branches = design["junctions"]
    groups = _junction_groups(design)
    cj = [sum(branches[index]["capacitance_F"] for index in group) * 1e15 for group in groups]
    ec, _ = reduce_islands(capacitance, np.empty((count, 0)), len(MODE_ORDER), cj)
    cell = linear_chain_cell(4, name=design["design_id"] + "-electrostatic")
    cell.update(parameter_origin="assumed_circuit" if synthetic else "extracted_full_cell", EC_matrix_GHz=ec.tolist(),
                EJ_idle_GHz=[sum(branches[index]["EJ_GHz"] for index in group) for group in groups],
                idle_flux_Phi0=[branches[group[0]]["bias_Phi0"] for group in groups],
                provenance={"design_sha256": digest(design), "geometry_extracted": not synthetic,
                            "synthetic": synthetic, "producer": "Synthetic fixture" if synthetic
                            else "Palace electrostatic terminated-island comparison"})
    if design["schema"] == ASSEMBLY_DESIGN_SCHEMA:
        cell["provenance"]["junction_capacitance_by_mode_fF"] = dict(zip(MODE_ORDER, cj, strict=True))
    return {"cell": cell, "source_sha256": files}


def _model(design, fine, train, scattering, qualification):
    ej = np.asarray([junction["EJ_GHz"] for junction in design["junctions"]])
    frequencies = fine["frequency_GHz"]
    p = fine["participation"]
    phase_zpf = np.sign(p) * np.sqrt(np.abs(p) * frequencies[:, None] / (2 * ej))
    phase_sq = phase_zpf**2
    cross_kerr = -(phase_sq * ej) @ phase_sq.T
    transitions = frequencies + 0.5 * cross_kerr.sum(axis=1)
    kerr = cross_kerr.copy()
    kerr[np.diag_indices_from(kerr)] *= 0.5
    if np.any(transitions <= 0):
        raise ValueError("Quartic approximation gives a nonpositive modal transition")
    model = {"schema": MODEL_SCHEMA, "design_sha256": digest(design), "qualification_sha256": digest(qualification),
             "mode_ids": fine["modes"].tolist(), "junction_order": [row["label"] for row in design["junctions"]],
             "ports": design["ports"],
             "linear_mode_GHz": frequencies.tolist(), "f01_GHz": transitions.tolist(),
             "zero_point_phase_rad": phase_zpf.tolist(), "kerr_GHz": kerr.tolist(),
             "kappa_signed_GHz": fine["kappa_signed_GHz"].tolist(), "frequency_grid_GHz": train.tolist(),
             "scattering_real": scattering.real.tolist(), "scattering_imag": scattering.imag.tolist(),
             "units": {"frequency": "GHz", "time": "ns", "phase": "rad", "scattering": "power wave ratio"},
             "approximation": "First-order quartic EPR at the fixed idle bias, diagonal in the modal Fock basis",
             "physical_device_validated": False, "synthetic": qualification["synthetic"]}
    if design["schema"] in (DESIGN_SCHEMA, ASSEMBLY_DESIGN_SCHEMA):
        model["readout_channels"] = design["readout_channels"]
    if design["design_id"] == SHARED_DESIGN_ID:
        model["readout_topology"] = "shared"
    if design["schema"] == ASSEMBLY_DESIGN_SCHEMA:
        model.update(readout_topology=design["readout_topology"],
                     junction_modes=[row["mode"] for row in design["junctions"]], junction_EJ_idle_GHz=ej.tolist(),
                     approximation="Fixed-idle-bias quartic EPR with one nonlinear contribution per junction branch",
                     omitted_effects=["Cubic terms of flux-biased junction branches"])
    return model


def qualify_reduction(coarse_dir, fine_dir, overlap_path, output, *, synthetic=False):
    """Save numeric qualification and bind a model only if every gate passes."""
    coarse_path, coarse_manifest, design = _case(coarse_dir, "coarse")
    fine_path, fine_manifest, fine_design = _case(fine_dir, "fine")
    if digest(design) != digest(fine_design) or coarse_manifest["files_sha256"]["mesh.msh"] == fine_manifest["files_sha256"]["mesh.msh"]:
        raise ValueError("Mesh comparison requires independent meshes of the same design and junction biases")
    coarse = _eigen(_stage_output(coarse_path, coarse_manifest, "eigenmode", synthetic=synthetic), design, synthetic=synthetic)
    fine = _eigen(_stage_output(fine_path, fine_manifest, "eigenmode", synthetic=synthetic), design, synthetic=synthetic)
    for path, eigen in ((coarse_path, coarse), (fine_path, fine)):
        if len(eigen["modes"]) != read_json(path / "eigenmode.json")["Solver"]["Eigenmode"]["N"]:
            raise ValueError("Palace eigenmode count differs from the frozen request")
    train, response, train_files = _scattering(_stage_output(fine_path, fine_manifest, "driven-train", synthetic=synthetic), design)
    holdout, holdout_response, holdout_files = _scattering(_stage_output(fine_path, fine_manifest, "driven-holdout", synthetic=synthetic), design)
    for label, grid in (("driven-train", train), ("driven-holdout", holdout)):
        samples = read_json(fine_path / (label + ".json"))["Solver"]["Driven"]["Samples"]
        planned_grid = np.asarray(samples[0]["Freq"]) if len(samples) == 1 and samples[0]["Type"] == "Point" else None
        if planned_grid is None or planned_grid.shape != grid.shape or not np.allclose(grid, planned_grid, rtol=0, atol=1e-8):
            raise ValueError(f"Palace data do not match the frozen {label} frequency grid")
    overlaps = _match_modes(read_json(overlap_path), coarse, fine)
    pairs = overlaps.pop("mode_pairs")
    eigen_metrics = {metric: max(coarse["metrics"][metric], fine["metrics"][metric]) for metric in
                     ("eigen_backward_error", "energy_relative", "junction_energy_fraction_abs", "loss_power_relative")}
    scattering_metrics, port_rms = _scattering_checks(train, response, holdout, holdout_response)
    metrics = {**eigen_metrics, **overlaps, **scattering_metrics}
    thresholds = TOLERANCE.copy()
    readout_errors = []
    if design["schema"] in (DESIGN_SCHEMA, ASSEMBLY_DESIGN_SCHEMA):
        readout_errors = _readout_through_errors(design, port_rms)
        metrics[READOUT_HOLDOUT_METRIC] = max(
            max(row["forward_heldout_complex_rms"], row["reverse_heldout_complex_rms"]) for row in readout_errors
        )
        thresholds[READOUT_HOLDOUT_METRIC] = TOLERANCE["heldout_complex_rms"]
    checks = {name: value >= thresholds[name] if name.endswith("_min") else value <= thresholds[name]
              for name, value in metrics.items()}
    checks["Q_consistency"] = all(case["metrics"]["Q_consistency"] for case in (coarse, fine))
    groups = _junction_groups(design)
    group_participation = np.column_stack([np.abs(fine["participation"][:, group]).sum(axis=1) for group in groups])
    junctions, mode_indices = linear_sum_assignment(group_participation.T, maximize=True)
    assignments = [{"junction": MODE_ORDER[j], "mode_id": int(fine["modes"][m]),
                    "frequency_GHz": float(fine["frequency_GHz"][m]),
                    "participation_abs": float(group_participation[m, j])} for j, m in zip(junctions, mode_indices)]
    if design["schema"] == ASSEMBLY_DESIGN_SCHEMA:
        for assignment, index in zip(assignments, junctions, strict=True):
            assignment["junction_branches"] = [design["junctions"][j]["label"] for j in groups[index]]
    checks["junction_modes_covered"] = len(assignments) == len(MODE_ORDER) and all(
        item["participation_abs"] >= .1 and train[0] < item["frequency_GHz"] < train[-1] for item in assignments
    )
    baseline = bind_electrostatic(_stage_output(fine_path, fine_manifest, "electrostatic", synthetic=synthetic),
                                  design, synthetic=synthetic)
    record = {"schema": QUALIFICATION_SCHEMA, "design_sha256": digest(design), "metrics": metrics,
              "thresholds": thresholds, "checks": checks, "passed": all(checks.values()), "mode_pairs": pairs,
              "junction_mode_assignments": assignments, "junction_participation_min": .1,
              "synthetic": synthetic, "coarse_input_sha256": coarse_manifest["files_sha256"],
              "fine_input_sha256": fine_manifest["files_sha256"], "overlap_sha256": file_hash(overlap_path),
              "coarse_fields_sha256": coarse["file_sha256"], "fine_fields_sha256": fine["file_sha256"],
              "training_sha256": train_files, "heldout_sha256": holdout_files,
              "electrostatic_sha256": baseline["source_sha256"], "physical_device_validated": False}
    if readout_errors:
        record["readout_through_interpolation"] = readout_errors
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "qualification.json", record)
    atomic_json(output / "electrostatic-cell.json", baseline["cell"])
    if record["passed"]:
        atomic_json(output / "model.json", _model(design, fine, train, response, record))
    manifest_names = ["qualification.json", "electrostatic-cell.json"]
    if record["passed"]:
        manifest_names.append("model.json")
    atomic_json(output / "artifact.json", {"schema": "emulateqpu.fullwave.artifact.v1", "passed": record["passed"],
                                          "files_sha256": {name: file_hash(output / name) for name in manifest_names}})
    return record
