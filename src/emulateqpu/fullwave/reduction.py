"""Read Palace field outputs and gate a separate, fixed-bias modal model."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..chain import linear_chain_cell
from ..io import atomic_json, digest, file_hash, read_json
from .design import DESIGN_SCHEMA, MODE_ORDER, PALACE_COMMIT, PALACE_TAG, PORT_ORDER

QUALIFICATION_SCHEMA = "emulateqpu.fullwave.qualification.v1"
MODEL_SCHEMA = "emulateqpu.fullwave.model.v1"
OVERLAP_SCHEMA = "emulateqpu.fullwave.mode-overlap.v1"
LOCAL_RUN_SCHEMA = "emulateqpu.fullwave.local-execution.v1"
STAGE_TABLES = {"electrostatic": ("terminal-C.csv",),
                "eigenmode": ("eig.csv", "port-EPR.csv", "port-Q.csv", "domain-E.csv"),
                "driven-train": ("port-S.csv",), "driven-holdout": ("port-S.csv",)}
TOLERANCE = {"eigen_backward_error": 1e-7, "mesh_frequency_relative": 1e-3,
             "mesh_mode_overlap_min": .90, "energy_relative": .01,
             "junction_energy_fraction_abs": .01, "loss_power_relative": .01, "scattering_passivity_abs": .01,
             "scattering_reciprocity_abs": .01, "heldout_complex_rms": .01}


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


def _case(directory, revision):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "field-inputs.json")
    design = read_json(directory / "design.json")
    if (manifest.get("schema") != "emulateqpu.fullwave.inputs.v1" or manifest.get("refinement") != revision
            or manifest.get("palace") != {"tag": PALACE_TAG, "commit": PALACE_COMMIT}
            or manifest.get("design_sha256") != digest(design)
            or design.get("schema") != DESIGN_SCHEMA or design.get("mode_order") != list(MODE_ORDER)
            or [port["label"] for port in design["ports"]] != list(PORT_ORDER)
            or [port["label"] for port in design["junctions"]] != list(MODE_ORDER)):
        raise ValueError("Four-data Palace input identity or ordering differs")
    expected = ("design.json", "mesh.msh", "electrostatic.json", "eigenmode.json", "driven-train.json", "driven-holdout.json")
    if set(manifest["files_sha256"]) != set(expected):
        raise ValueError("Incomplete frozen Palace input inventory")
    for file, expected_hash in manifest["files_sha256"].items():
        path = directory / file
        if path.is_symlink() or file_hash(path) != expected_hash:
            raise ValueError(f"Palace input integrity mismatch for {file}")
    return directory, manifest, design


def _stage_output(case, manifest, stage, *, synthetic=False):
    native = case / f"{stage}-output"
    batch = case / f"{stage}-slurm"
    if batch.exists():
        from ..field_slurm import collect

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
    else:
        stages = native
        if not synthetic:
            receipt = read_json(stages / "fullwave-execution.json")
            hashes = manifest["files_sha256"]
            evidence = _field_provenance(stages, STAGE_TABLES[stage], solved_mesh=stage == "eigenmode", synthetic=False)
            if (receipt.get("schema") != LOCAL_RUN_SCHEMA or receipt.get("stage") != stage
                    or receipt.get("config_sha256") != hashes[f"{stage}.json"]
                    or receipt.get("mesh_sha256") != hashes["mesh.msh"]
                    or receipt.get("execution_sha256") != file_hash(stages / "execution.json")
                    or receipt.get("field_sha256") != evidence
                    or read_json(stages / "execution.json").get("status") != "COMPLETE"):
                raise ValueError(f"Local output is stale, incomplete or lacks a bound launch for {stage}")
    version = read_json(stages / "palace.json")
    if version.get("Problem", {}).get("MeshElements") != manifest["tetrahedra"]:
        raise ValueError(f"Palace solved mesh cell count differs from the input manifest for {stage}")
    return stages


def solve_local(case, stage, *, executable, memory_mb=4096, wall_time_s=600):
    """Run a single-rank Palace stage from frozen inputs and register its output evidence."""
    from ..palace import launch_palace, plan_palace

    if stage not in STAGE_TABLES:
        raise ValueError("Unknown full-wave stage")
    revision = read_json(Path(case) / "field-inputs.json")["refinement"]
    case, manifest, _ = _case(case, revision)
    hashes = manifest["files_sha256"]
    config = case / f"{stage}.json"
    output = case / f"{stage}-output"
    if (case / f"{stage}-slurm").exists():
        raise ValueError(f"A Slurm run already exists for {stage}; use a fresh case")
    plan = plan_palace(config, executable=executable, memory_mb=memory_mb, wall_time_s=wall_time_s)
    if plan["config_sha256"] != hashes[config.name]:
        raise ValueError("Palace configuration changed before launch")
    execution = launch_palace(plan, output)
    if execution["status"] != "COMPLETE":
        return execution
    if file_hash(case / "mesh.msh") != hashes["mesh.msh"]:
        raise ValueError("Palace mesh changed during launch")
    evidence = _field_provenance(output, STAGE_TABLES[stage], solved_mesh=stage == "eigenmode", synthetic=False)
    receipt = {"schema": LOCAL_RUN_SCHEMA, "stage": stage, "status": "COMPLETE",
               "config_sha256": hashes[config.name], "mesh_sha256": hashes["mesh.msh"],
               "execution_sha256": file_hash(output / "execution.json"), "field_sha256": evidence}
    atomic_json(output / "fullwave-execution.json", receipt)
    return receipt


def _field_provenance(directory, files, *, solved_mesh, synthetic):
    version = read_json(directory / "palace.json")
    expected_tags = {PALACE_TAG, PALACE_TAG.removeprefix("v"), PALACE_COMMIT, PALACE_COMMIT[:7]}
    info = version.get("Problem", {})
    if (version.get("GitTag") not in expected_tags or type(info.get("MPISize")) is not int
            or info["MPISize"] < 1 or type(info.get("MeshElements")) is not int or info["MeshElements"] < 1):
        raise ValueError(f"Missing pinned Palace build and run metadata in {directory}")
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
    full_response = np.concatenate((train_response, holdout_response))
    return {"scattering_passivity_abs": float(max(0, np.linalg.svd(full_response, compute_uv=False).max() - 1)),
            "scattering_reciprocity_abs": float(np.max(np.abs(full_response - full_response.transpose(0, 2, 1)))),
            "heldout_complex_rms": float(np.sqrt(np.mean(np.abs(predicted - holdout_response)**2)))}


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
    ec, _ = reduce_islands(capacitance, np.empty((count, 0)), len(MODE_ORDER),
                           [junction["capacitance_F"] * 1e15 for junction in design["junctions"]])
    cell = linear_chain_cell(4, name=design["design_id"] + "-electrostatic")
    cell.update(parameter_origin="assumed_circuit" if synthetic else "extracted_full_cell", EC_matrix_GHz=ec.tolist(),
                EJ_idle_GHz=[junction["EJ_GHz"] for junction in design["junctions"]],
                idle_flux_Phi0=[junction["bias_Phi0"] for junction in design["junctions"]],
                provenance={"design_sha256": digest(design), "geometry_extracted": not synthetic,
                            "synthetic": synthetic, "producer": "Synthetic fixture" if synthetic
                            else "Palace electrostatic terminated-island comparison"})
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
    return {"schema": MODEL_SCHEMA, "design_sha256": digest(design), "qualification_sha256": digest(qualification),
            "mode_ids": fine["modes"].tolist(), "junction_order": list(MODE_ORDER), "ports": design["ports"],
            "linear_mode_GHz": frequencies.tolist(), "f01_GHz": transitions.tolist(),
            "zero_point_phase_rad": phase_zpf.tolist(), "kerr_GHz": kerr.tolist(),
            "kappa_signed_GHz": fine["kappa_signed_GHz"].tolist(), "frequency_grid_GHz": train.tolist(),
            "scattering_real": scattering.real.tolist(), "scattering_imag": scattering.imag.tolist(),
            "units": {"frequency": "GHz", "time": "ns", "phase": "rad", "scattering": "power wave ratio"},
            "approximation": "First-order quartic EPR at the fixed idle bias, diagonal in the modal Fock basis",
            "physical_device_validated": False, "synthetic": qualification["synthetic"]}


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
    metrics = {**eigen_metrics, **overlaps, **_scattering_checks(train, response, holdout, holdout_response)}
    checks = {name: value >= TOLERANCE[name] if name.endswith("_min") else value <= TOLERANCE[name]
              for name, value in metrics.items()}
    checks["Q_consistency"] = all(case["metrics"]["Q_consistency"] for case in (coarse, fine))
    junctions, mode_indices = linear_sum_assignment(np.abs(fine["participation"].T), maximize=True)
    assignments = [{"junction": design["junctions"][j]["label"], "mode_id": int(fine["modes"][m]),
                    "frequency_GHz": float(fine["frequency_GHz"][m]),
                    "participation_abs": float(abs(fine["participation"][m, j]))} for j, m in zip(junctions, mode_indices)]
    checks["junction_modes_covered"] = len(assignments) == len(MODE_ORDER) and all(
        item["participation_abs"] >= .1 and train[0] < item["frequency_GHz"] < train[-1] for item in assignments
    )
    baseline = bind_electrostatic(_stage_output(fine_path, fine_manifest, "electrostatic", synthetic=synthetic),
                                  design, synthetic=synthetic)
    record = {"schema": QUALIFICATION_SCHEMA, "design_sha256": digest(design), "metrics": metrics,
              "thresholds": TOLERANCE, "checks": checks, "passed": all(checks.values()), "mode_pairs": pairs,
              "junction_mode_assignments": assignments, "junction_participation_min": .1,
              "synthetic": synthetic, "coarse_input_sha256": coarse_manifest["files_sha256"],
              "fine_input_sha256": fine_manifest["files_sha256"], "overlap_sha256": file_hash(overlap_path),
              "coarse_fields_sha256": coarse["file_sha256"], "fine_fields_sha256": fine["file_sha256"],
              "training_sha256": train_files, "heldout_sha256": holdout_files,
              "electrostatic_sha256": baseline["source_sha256"], "physical_device_validated": False}
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
