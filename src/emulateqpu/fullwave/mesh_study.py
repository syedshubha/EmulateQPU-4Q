"""Measure fixed-geometry electrostatic mesh convergence for the four-qubit chip."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from ..io import atomic_json, file_hash, read_json
from ..palace import launch_palace, plan_palace
from .design import (LADDER_MESHES, LADDER_OUTPUT_DIRECTORY, PALACE_COMMIT, PALACE_TAG, _configs,
                     _electrostatic_options, ladder_mesh_settings, prepare_design)
from .reduction import _case, _column, _indices, _table

STUDY_SCHEMA = "emulateqpu.fullwave.electrostatic-mesh-study.v1"
REPORT_SCHEMA = "emulateqpu.fullwave.electrostatic-mesh-report.v1"
THRESHOLDS = {"terminal_C_relative_frobenius": 0.02,
              "differential_EC_relative_frobenius": 0.02,
              "differential_EC_diagonal_max_relative": 0.02}
MATRIX_TOLERANCES = {"C_reciprocity_relative_max": 1e-6, "C_positive_mutual_max_F": 1e-25}
PLAN_FILE = "mesh-study.json"
SOURCE_TAGS = {PALACE_TAG, PALACE_TAG.removeprefix("v"), PALACE_COMMIT, PALACE_COMMIT[:7]}
OUTPUT_FILES = ("palace.json", "terminal-C.csv", "launch_plan.json", "execution.json", "stdout.log", "stderr.log")
CRACK_BISECTION = re.compile(r"Added (\d+) elements in (\d+) iterations of local bisection for "
                             r"under-resolved interior boundaries")
NONCONVERGENCE = re.compile(r"\b(did not converge|failed to converge|not converged)\b", re.I)


def _file(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or unsafe mesh study file {path}")
    return path


def _input_bindings(root, order, full_assembly):
    if type(order) is not int or order not in (1, 2):
        raise ValueError("Electrostatic order must be 1 or 2")
    common_design, bindings = None, []
    for level in LADDER_MESHES:
        case = root / level
        if case.is_symlink():
            raise ValueError(f"Unsafe mesh level {case}")
        _, manifest, design = _case(case, level)
        settings = ladder_mesh_settings(level)
        count = manifest["tetrahedra"]
        if (manifest["mesh_settings"] != settings or type(count) is not int
                or not 0 < count <= settings["maximum_tetrahedra"]):
            raise ValueError(f"Mesh settings or tetrahedron count differ for {level}")
        static = _configs(design, manifest["physical_surface_ids"], manifest["material_ids"], 12, [], [])["electrostatic"]
        _electrostatic_options(static, level, order, full_assembly)
        config = read_json(case / "electrostatic.json")
        if config != static:
            raise ValueError(f"Electrostatic configuration differs from the fixed study template for {level}")
        signature = (manifest["design_sha256"], static)
        if common_design is not None and common_design != signature:
            raise ValueError("Mesh study requires identical geometry, terminal order and solver settings")
        common_design = signature
        bindings.append({"level": level, "tetrahedra": count, "mesh_settings": settings,
                         "field_inputs_sha256": file_hash(_file(case / "field-inputs.json")),
                         "mesh_sha256": manifest["files_sha256"]["mesh.msh"],
                         "config_sha256": manifest["files_sha256"]["electrostatic.json"]})
    if (len({row["mesh_sha256"] for row in bindings}) != len(bindings)
            or any(a["tetrahedra"] >= b["tetrahedra"] for a, b in zip(bindings, bindings[1:]))):
        raise ValueError("Mesh ladder requires distinct meshes with strictly increasing tetrahedron counts")
    return common_design[0], bindings


def register_study(root, *, order=2, full_assembly=False):
    """Freeze existing ladder inputs and acceptance criteria, preserving any partial outputs."""
    root = Path(root).resolve()
    if (root / PLAN_FILE).exists():
        raise FileExistsError(root / PLAN_FILE)
    if type(full_assembly) is not bool:
        raise ValueError("Electrostatic full_assembly must be a boolean")
    design_sha, bindings = _input_bindings(root, order, full_assembly)
    plan = {"schema": STUDY_SCHEMA, "design_sha256": design_sha, "electrostatic_order": order,
            "levels": bindings, "thresholds": THRESHOLDS, "matrix_tolerances": MATRIX_TOLERANCES,
            "posthoc_setup": any((root / level / LADDER_OUTPUT_DIRECTORY).exists() for level in LADDER_MESHES)}
    if full_assembly:
        plan["full_assembly"] = True
    atomic_json(root / PLAN_FILE, plan)
    return plan


def prepare_study(root, *, order=2, full_assembly=False):
    """Generate three fixed-geometry meshes and register the study before any solve."""
    if type(order) is not int or order not in (1, 2):
        raise ValueError("Electrostatic order must be 1 or 2")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for level in LADDER_MESHES:
        prepare_design(root / level, refinement=level, electrostatic_order=order,
                       electrostatic_full_assembly=full_assembly)
    return register_study(root, order=order, full_assembly=full_assembly)


def _study(root):
    plan_file = _file(root / PLAN_FILE)
    plan = read_json(plan_file)
    if (plan["schema"] != STUDY_SCHEMA or plan["thresholds"] != THRESHOLDS
            or plan["matrix_tolerances"] != MATRIX_TOLERANCES or type(plan["posthoc_setup"]) is not bool):
        raise ValueError("Mesh study schema or predeclared acceptance criteria differ")
    full_assembly = plan.get("full_assembly", False)
    if type(full_assembly) is not bool:
        raise ValueError("Invalid mesh study assembly choice")
    design_sha, bindings = _input_bindings(root, plan["electrostatic_order"], full_assembly)
    if design_sha != plan["design_sha256"] or bindings != plan["levels"]:
        raise ValueError("Mesh inputs changed since study registration")
    return plan, file_hash(plan_file)


def _source_pinned(plan):
    identity = plan["executable_identity"]
    build = plan["supplied_build_manifest"]
    return (identity["kind"] == "direct_binary_or_unknown" and build.get("source_commit") == PALACE_COMMIT
            and build.get("executable_sha256") == identity["executable_sha256"])


def _run_binding(study_sha, level, plan, diagnostic):
    return {"study_sha256": study_sha, "mesh_sha256": level["mesh_sha256"],
            "config_sha256": level["config_sha256"], "palace_plan": plan, "diagnostic": diagnostic}


def run_level(root, level_name, *, executable, mpi_ranks=1, launcher=None, build_manifest=None,
              memory_mb=8192, wall_time_s=600, diagnostic=False):
    """Run one mesh level locally and bind input and output hashes to its native launch."""
    root = Path(root).resolve()
    study, study_sha = _study(root)
    level = next((row for row in study["levels"] if row["level"] == level_name), None)
    if level is None:
        raise ValueError(f"Unknown mesh level {level_name!r}")
    case = root / level_name
    raw = case / LADDER_OUTPUT_DIRECTORY
    if raw.exists():
        raise FileExistsError(raw)
    plan = plan_palace(case / "electrostatic.json", executable=executable, mpi_ranks=mpi_ranks,
                       launcher=launcher, build_manifest=build_manifest,
                       memory_mb=memory_mb, wall_time_s=wall_time_s)
    if plan["config_sha256"] != level["config_sha256"] or file_hash(case / "mesh.msh") != level["mesh_sha256"]:
        raise ValueError("Mesh inputs changed before Palace launch")
    if not diagnostic and (study["posthoc_setup"] or study["electrostatic_order"] != 2 or not _source_pinned(plan)):
        raise ValueError("Qualification requires preregistered P2 inputs and the direct source-pinned Palace build")
    receipt = case / "mesh-study-run"
    receipt.mkdir()
    atomic_json(receipt / "before.json", _run_binding(study_sha, level, plan, diagnostic))
    execution = launch_palace(plan, raw)
    if execution["status"] != "COMPLETE":
        return execution
    if (_study(root)[1] != study_sha or file_hash(plan["executable_identity"]["executable"])
            != plan["executable_identity"]["executable_sha256"]):
        raise ValueError("Mesh study inputs or Palace executable changed during launch")
    output = {"status": "COMPLETE", "before_sha256": file_hash(receipt / "before.json"),
              "output_sha256": _output_hashes(raw)}
    atomic_json(receipt / "after.json", output)
    return output


def _output_hashes(raw):
    return {name: file_hash(_file(raw / name)) for name in OUTPUT_FILES}


def _native_preprocessing(raw):
    """Read Palace's geometric crack repair, independently of solution adaptivity."""
    bisection = []
    for name in ("stdout.log", "stderr.log"):
        with (raw / name).open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if NONCONVERGENCE.search(line):
                    raise ValueError("Palace reported a nonconverged solve")
                match = CRACK_BISECTION.search(line)
                if match:
                    bisection.append(tuple(map(int, match.groups())))
    if len(bisection) > 1 or bisection and min(bisection[0]) < 1:
        raise ValueError("Ambiguous Palace geometric bisection record")
    added, iterations = bisection[0] if bisection else (0, 0)
    return {"added_tetrahedra": added, "local_bisection_iterations": iterations}


def _native_run(case, level, study_sha, design, diagnostic_unbound):
    raw = case / LADDER_OUTPUT_DIRECTORY
    if raw.is_symlink():
        raise ValueError("Unsafe Palace output directory")
    if not raw.exists():
        return {"status": "PENDING"}, None
    execution_path = raw / "execution.json"
    if not execution_path.exists():
        return {"status": "INCOMPLETE", "reason": "Missing Palace execution record"}, None
    execution = read_json(_file(execution_path))
    if execution["schema"] != "emulateqpu.palace_execution.v1":
        raise ValueError("Unexpected Palace execution schema")
    if execution["status"] != "COMPLETE":
        return {"status": "FAILED", "reason": execution["resource_stop"] or f"Exit {execution['returncode']}"}, None
    plan = read_json(_file(raw / "launch_plan.json"))
    if (plan != execution["plan"] or execution["returncode"] != 0 or execution["resource_stop"] is not None
            or plan["schema"] != "emulateqpu.palace_plan.v1" or execution["command"] != plan["command"]
            or Path(plan["config"]).name != "electrostatic.json"
            or plan["config_sha256"] != level["config_sha256"] or plan["device"] != "CPU"):
        raise ValueError("Palace execution differs from the recorded mesh study launch")
    receipt = case / "mesh-study-run"
    hashes = _output_hashes(raw)
    if receipt.exists():
        if receipt.is_symlink():
            raise ValueError("Unsafe mesh study receipt directory")
        before_path = _file(receipt / "before.json")
        before = read_json(before_path)
        after = read_json(_file(receipt / "after.json"))
        if (type(before["diagnostic"]) is not bool
                or before != _run_binding(study_sha, level, plan, before["diagnostic"])
                or after != {"status": "COMPLETE", "before_sha256": file_hash(before_path), "output_sha256": hashes}):
            raise ValueError("Mesh study input or output receipt failed its SHA-256 check")
        bound = not before["diagnostic"] and _source_pinned(plan)
    elif diagnostic_unbound:
        bound = False
    else:
        return {"status": "UNVERIFIED", "reason": "Missing pre-launch and post-launch mesh study receipts"}, None
    native = read_json(raw / "palace.json")
    problem = native["Problem"]
    tag, ranks, solved_cells = native["GitTag"], problem["MPISize"], problem["MeshElements"]
    preprocessing = _native_preprocessing(raw)
    if (not isinstance(tag, str) or not tag or (bound and tag not in SOURCE_TAGS)
            or type(ranks) is not int or ranks < 1 or ranks != plan["mpi_ranks"]
            or type(solved_cells) is not int or solved_cells != level["tetrahedra"] + preprocessing["added_tetrahedra"]
            or solved_cells > level["mesh_settings"]["maximum_tetrahedra"]
            or native["ElapsedTime"]["Counts"]["Adaptation"] != 0
            or native["LinearSolver"]["TotalSolves"] != len(design["island_order"])):
        raise ValueError("Palace source, rank count, mesh or solve inventory differs from the study")
    identity = plan["executable_identity"]
    run_identity = {"GitTag": tag, "executable_sha256": identity["executable_sha256"],
                    "sibling_binary_sha256": sorted(item["sha256"] for item in identity["sibling_binary_candidates"]),
                    "mpi_ranks": ranks, "threads_per_rank": plan["threads_per_rank"]}
    matrices, metrics = _matrices(raw / "terminal-C.csv", design)
    return {"status": "COMPLETE", "source_pinned_and_bound": bound, "run_identity": run_identity,
            "raw_sha256": hashes, "elapsed_s": execution["elapsed_s"],
            "observed_peak_rss_bytes": execution["observed_peak_rss_bytes"],
            "solved_tetrahedra": solved_cells, "native_mesh_preprocessing": preprocessing,
            "matrix_checks": metrics, "terminal_C_F": matrices[0].tolist(),
            "differential_EC_GHz": matrices[1].tolist()}, matrices


def _matrices(path, design):
    from .._physics.chain_family import reduce_islands

    count = len(design["island_order"])
    table = _table(path)
    columns = [f"C[i][{i}] (F)" for i in range(1, count + 1)]
    if set(table) != {"i", *columns} or not np.array_equal(_indices(table, "i"), np.arange(1, count + 1)):
        raise ValueError("Capacitance terminal inventory differs from the design")
    capacitance = np.column_stack([_column(table, name) for name in columns])
    diagonal = np.diag(capacitance)
    if np.min(diagonal) <= 0:
        raise ValueError("Nonpositive terminal self-capacitance")
    reciprocal_error = float(np.max(np.abs(capacitance - capacitance.T)) / np.max(np.abs(capacitance)))
    mutual_max = float(np.max(capacitance - np.diag(diagonal)))
    minimum_eigenvalue = float(np.linalg.eigvalsh((capacitance + capacitance.T) / 2)[0])
    if (reciprocal_error > MATRIX_TOLERANCES["C_reciprocity_relative_max"]
            or mutual_max > MATRIX_TOLERANCES["C_positive_mutual_max_F"] or minimum_eigenvalue <= 0):
        raise ValueError("Terminal capacitance matrix failed reciprocity, mutual sign or positive definiteness")
    ec, _ = reduce_islands(capacitance, np.empty((count, 0)), len(design["mode_order"]),
                           [row["capacitance_F"] * 1e15 for row in design["junctions"]])
    if not np.isfinite(ec).all() or np.linalg.eigvalsh(ec)[0] <= 0:
        raise ValueError("Differential charging-energy matrix failed finite positive definiteness")
    return (capacitance, ec), {"C_reciprocity_relative_max": reciprocal_error,
                              "C_positive_mutual_max_F": mutual_max, "C_min_eigenvalue_F": minimum_eigenvalue,
                              "EC_min_eigenvalue_GHz": float(np.linalg.eigvalsh(ec)[0])}


def _interval(coarse, fine, modes):
    c0, ec0 = coarse
    c1, ec1 = fine
    delta_c, delta_ec = c1 - c0, ec1 - ec0
    diag_ec = np.abs(np.diag(delta_ec) / np.diag(ec1))
    metrics = {"terminal_C_relative_frobenius": float(np.linalg.norm(delta_c) / np.linalg.norm(c1)),
               "differential_EC_relative_frobenius": float(np.linalg.norm(delta_ec) / np.linalg.norm(ec1)),
               "differential_EC_diagonal_max_relative": float(diag_ec.max())}
    neighbors = [{"modes": [a, b], "coarse_MHz": float(ec0[i, i + 1] * 1000),
                  "fine_MHz": float(ec1[i, i + 1] * 1000), "delta_MHz": float(delta_ec[i, i + 1] * 1000)}
                 for i, (a, b) in enumerate(zip(modes, modes[1:]))]
    return {"metrics": metrics, "passed": all(metrics[key] <= limit for key, limit in THRESHOLDS.items()),
            "terminal_C_max_absolute_delta_fF": float(np.max(np.abs(delta_c)) * 1e15),
            "differential_EC_diagonal_relative_by_mode": dict(zip(modes, diag_ec.tolist())),
            "differential_EC_nearest_neighbors": neighbors,
            "differential_EC_nearest_neighbor_max_absolute_delta_MHz": max(abs(pair["delta_MHz"]) for pair in neighbors)}


def analyze_study(root, output, *, diagnostic_unbound=False):
    """Write a complete, failed or partial report into a fresh results directory."""
    root = Path(root).resolve()
    study, study_sha = _study(root)
    design = read_json(root / study["levels"][0]["level"] / "design.json")
    levels, matrices = {}, {}
    for level in study["levels"]:
        name = level["level"]
        try:
            progress, matrix = _native_run(root / name, level, study_sha, design, diagnostic_unbound)
        except (OSError, ValueError, KeyError, TypeError) as error:
            progress, matrix = {"status": "INVALID", "reason": str(error)}, None
        levels[name] = {**level, **progress}
        if matrix is not None:
            matrices[name] = matrix
    names = tuple(LADDER_MESHES)
    identities = [levels[name]["run_identity"] for name in matrices]
    consistent = bool(identities) and all(identity == identities[0] for identity in identities)
    solved_counts = [levels[name]["solved_tetrahedra"] for name in matrices]
    increasing = all(left < right for left, right in zip(solved_counts, solved_counts[1:]))
    intervals = []
    for coarse, fine in zip(names, names[1:]):
        if coarse in matrices and fine in matrices and levels[coarse]["run_identity"] == levels[fine]["run_identity"]:
            intervals.append({"coarse": coarse, "fine": fine,
                              **_interval(matrices[coarse], matrices[fine], design["mode_order"])})
    complete = len(matrices) == len(names) and consistent and increasing
    passed = complete and all(interval["passed"] for interval in intervals)
    source_bound = (complete and not study["posthoc_setup"] and study["electrostatic_order"] == 2
                    and all(levels[name]["source_pinned_and_bound"] for name in names))
    if complete:
        status = "CONVERGED" if passed else "NOT_CONVERGED"
        if not source_bound:
            status = f"DIAGNOSTIC_{status}"
    elif increasing:
        status = "INCOMPLETE" if consistent or not matrices else "INCONSISTENT_EXECUTION"
    else:
        status = "INVALID_SOLVED_MESH_ORDER"
    report = {"schema": REPORT_SCHEMA, "status": status, "screen_passed": passed,
              "electrostatic_mesh_qualified": source_bound and passed, "study_sha256": study_sha,
              "design_sha256": study["design_sha256"], "electrostatic_order": study["electrostatic_order"],
              "full_assembly": study.get("full_assembly", False),
              "thresholds": THRESHOLDS, "matrix_tolerances": MATRIX_TOLERANCES, "levels": list(levels.values()),
              "terminal_order": design["island_order"], "mode_order": design["mode_order"], "intervals": intervals}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "mesh-report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "register"):
        command = commands.add_parser(name)
        command.add_argument("study", type=Path)
        command.add_argument("--order", type=int, choices=(1, 2), default=2)
        command.add_argument("--full-assembly", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("study", type=Path)
    run.add_argument("level", choices=tuple(LADDER_MESHES))
    run.add_argument("--executable", required=True)
    run.add_argument("--mpi-ranks", type=int, default=1)
    run.add_argument("--launcher", choices=("mpirun", "mpiexec", "srun"))
    run.add_argument("--build-manifest", type=Path)
    run.add_argument("--memory-mb", type=int, default=8192)
    run.add_argument("--wall-time-s", type=int, default=600)
    run.add_argument("--diagnostic", action="store_true")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("study", type=Path)
    analyze.add_argument("output", type=Path)
    analyze.add_argument("--diagnostic-unbound", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_study(args.study, order=args.order, full_assembly=args.full_assembly)
        elif args.command == "register":
            result = register_study(args.study, order=args.order, full_assembly=args.full_assembly)
        elif args.command == "run":
            result = run_level(args.study, args.level, executable=args.executable, mpi_ranks=args.mpi_ranks,
                               launcher=args.launcher, build_manifest=args.build_manifest, memory_mb=args.memory_mb,
                               wall_time_s=args.wall_time_s, diagnostic=args.diagnostic)
        else:
            result = analyze_study(args.study, args.output, diagnostic_unbound=args.diagnostic_unbound)
    except (ImportError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("screen_passed", result.get("status") != "FAILED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
