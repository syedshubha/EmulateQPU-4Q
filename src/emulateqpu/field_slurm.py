"""Render and verify CPU Palace field jobs on a shared Slurm filesystem."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from . import hpc
from .execution import THREAD_VARIABLES, _file_manifest, _now
from .io import atomic_json, file_hash, read_json

PROFILE_SCHEMA = "emulateqpu.field_slurm_profile.v1"
PLAN_SCHEMA = "emulateqpu.field_slurm_plan.v1"
RUN_SCHEMA = "emulateqpu.field_slurm_run.v1"
EXECUTION_SCHEMA = "emulateqpu.field_slurm_execution.v1"
SUBMISSION_SCHEMA = "emulateqpu.field_slurm_submission.v1"
COLLECTION_SCHEMA = "emulateqpu.field_slurm_collection.v1"
STATUS_SCHEMA = "emulateqpu.field_slurm_status.v1"
PALACE_LOG_DIRECTORY = "palace-logs"
PALACE_LOG_NAMES = ("stdout.log", "stderr.log")

SITE_KEYS = {"shared_filesystem", "shared_root", "python", "partition", "account", "qos",
             "job_name", "wall_time", "memory_mb", "environment"}
FIELD_KEYS = {"schema", "palace_binary", "build_manifest", "nodes", "mpi_ranks", "mpi_type"}
OUTPUTS = {"Electrostatic": "terminal-C.csv", "Eigenmode": "eig.csv", "Driven": "port-S.csv",
           "Magnetostatic": "surface-F.csv"}


def _read(value):
    return read_json(value) if isinstance(value, (str, Path)) else value


def _absolute_path(value, field):
    if not isinstance(value, str) or not Path(value).is_absolute() or any(c in value for c in "\0\r\n"):
        raise ValueError(f"{field} must be an absolute path on the cluster")
    return Path(value)


def _in_shared_root(path, profile):
    if not path.is_relative_to(Path(profile["shared_root"])):
        raise ValueError(f"Path is outside the declared shared filesystem: {path}")
    return path


def validate_profile(value):
    value = _read(value)
    if not isinstance(value, dict) or value.get("schema") != PROFILE_SCHEMA:
        raise ValueError("Expected a Palace Slurm site profile")
    if set(value) - SITE_KEYS - FIELD_KEYS:
        raise ValueError("Unknown Palace Slurm site profile fields")
    required = {"shared_root", "python", "palace_binary", "build_manifest", "wall_time", "memory_mb"}
    if missing := sorted(required - set(value)):
        raise ValueError("Missing Palace Slurm site profile fields: " + ", ".join(missing))
    profile = hpc.validate_profile({key: value[key] for key in SITE_KEYS & value.keys()})
    profile = {key: profile[key] for key in SITE_KEYS if key in profile}
    profile["schema"] = PROFILE_SCHEMA
    for field in ("shared_root", "python", "palace_binary", "build_manifest"):
        location = _absolute_path(value[field], field)
        # Resolving the interpreter link would bypass a shared virtual environment.
        profile[field] = str(location if field == "python" else location.resolve())
    for field, maximum in (("nodes", 256), ("mpi_ranks", 4096)):
        number = value.get(field, 1)
        if type(number) is not int or not 1 <= number <= maximum:
            raise ValueError(f"{field} must be an integer in 1..{maximum}")
        profile[field] = number
    if profile["nodes"] > profile["mpi_ranks"]:
        raise ValueError("Palace requires at least one MPI rank per allocated node")
    if "mpi_type" in value:
        if not isinstance(value["mpi_type"], str) or not re.fullmatch(r"[A-Za-z0-9_+.-]+", value["mpi_type"]):
            raise ValueError("Invalid site MPI type")
        profile["mpi_type"] = value["mpi_type"]
    if any(key.startswith(("SLURM_", "PMI_", "PMIX_", "OMPI_COMM_WORLD_"))
           for key in profile["environment"]):
        raise ValueError("Field profiles cannot override Slurm or MPI allocation variables")
    return profile


def _fingerprint(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Required field input is missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def _verify_inputs(value):
    for entry in (value["config"], value["binary"], value["build"], *value["inputs"]):
        if _fingerprint(entry["path"]) != entry:
            raise ValueError(f"Field inputs changed since planning: {entry['path']}")


def _tables(config, required_outputs):
    kind = config["Problem"]["Type"]
    controls = config["Solver"].get(kind, {})
    if not isinstance(controls, dict):
        raise ValueError(f"Palace Solver.{kind} must be an object")
    lower_bound = controls.get("N", 1) if kind == "Eigenmode" else 1
    if type(lower_bound) is not int or lower_bound < 1:
        raise ValueError("Palace eigenmode count must be a positive integer")
    if kind == "Driven":
        points = set()
        for sample in controls.get("Samples", []):
            if isinstance(sample, dict) and sample.get("Type") == "Point":
                frequencies = sample.get("Freq")
                if not isinstance(frequencies, list) or not all(
                    type(value) in (int, float) and math.isfinite(value) and value > 0 for value in frequencies
                ):
                    raise ValueError("Palace Point samples must specify positive frequencies")
                points.update(frequencies)
        lower_bound = max(1, len(points))
    elif kind == "Electrostatic":
        boundaries = config.get("Boundaries", {})
        terminals = boundaries.get("Terminal") if isinstance(boundaries, dict) else None
        if not isinstance(terminals, list) or not terminals:
            raise ValueError("Palace electrostatic jobs require at least one declared terminal")
        lower_bound = len(terminals)
    elif kind == "Magnetostatic":
        boundaries = config.get("Boundaries", {})
        sources = boundaries.get("SurfaceCurrent") if isinstance(boundaries, dict) else None
        postprocessing = boundaries.get("Postprocessing", {}) if isinstance(boundaries, dict) else None
        pickups = postprocessing.get("SurfaceFlux") if isinstance(postprocessing, dict) else None
        if not isinstance(sources, list) or not sources or not isinstance(pickups, list) or not pickups:
            raise ValueError("Magnetostatic field jobs require surface-current sources and flux pickups")
        lower_bound = len(sources)
    tables = {OUTPUTS[kind]: lower_bound}
    if kind == "Magnetostatic":
        tables["terminal-I.csv"] = lower_bound
    for name, count in (required_outputs or {}).items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.csv", name) or type(count) is not int or count < 1:
            raise ValueError("Expected output tables must identify CSV names and positive minimum row counts")
        tables[name] = max(count, tables.get(name, 0))
    return tables


def plan(config, profile, *, inputs=(), required_outputs=None):
    """Fingerprint inputs and renderable settings without invoking Palace or Slurm."""
    profile = validate_profile(profile)
    config = _in_shared_root(Path(config).expanduser().resolve(), profile)
    source = _fingerprint(config)
    settings = read_json(config)
    if not isinstance(settings, dict) or any(not isinstance(settings.get(key), dict)
                                             for key in ("Problem", "Model", "Solver")):
        raise ValueError("Palace settings require Problem, Model, and Solver objects")
    kind = settings["Problem"].get("Type")
    if not isinstance(kind, str) or kind not in OUTPUTS:
        raise ValueError("Supported Palace field problem types: " + ", ".join(OUTPUTS))
    if settings.get("Solver", {}).get("Device", "CPU") != "CPU":
        raise ValueError("This field Slurm workflow requires a CPU Palace configuration")
    mesh_setting = settings.get("Model", {}).get("Mesh")
    if not isinstance(mesh_setting, str) or not mesh_setting:
        raise ValueError("Palace Model.Mesh must identify the input mesh")
    mesh = _in_shared_root((config.parent / mesh_setting).resolve(), profile)
    dependencies = {mesh, *(_in_shared_root((config.parent / path).resolve(), profile) for path in inputs)}
    binary = _fingerprint(profile["palace_binary"])
    with Path(binary["path"]).open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("palace_binary must identify the direct Linux Palace executable")
    build = _fingerprint(profile["build_manifest"])
    build_manifest = read_json(build["path"])
    if not isinstance(build_manifest, dict) or build_manifest.get("executable_sha256") != binary["sha256"]:
        raise ValueError("Palace build manifest does not match the direct executable")
    return {"schema": PLAN_SCHEMA, "config": source, "binary": binary, "build": build,
            "mesh": str(mesh), "inputs": [_fingerprint(path) for path in sorted(dependencies)],
            "required_outputs": _tables(settings, required_outputs), "profile": profile,
            "field_numerical_qualification": "NOT_RUN"}


def _command(profile, config):
    command = ["srun", f"--nodes={profile['nodes']}", f"--ntasks={profile['mpi_ranks']}",
               "--cpus-per-task=1", "--kill-on-bad-exit=1"]
    if "mpi_type" in profile:
        command.append("--mpi=" + profile["mpi_type"])
    return [*command, profile["palace_binary"], str(config)]


def render(value, output):
    """Stage the run and emit a batch script without calling a scheduler."""
    value = _read(value)
    if not isinstance(value, dict) or value.get("schema") != PLAN_SCHEMA:
        raise ValueError("Expected a Palace Slurm field plan")
    profile = validate_profile(value["profile"])
    canonical = plan(value["config"]["path"], profile,
                     inputs=[entry["path"] for entry in value["inputs"]],
                     required_outputs=value["required_outputs"])
    if canonical != value:
        raise ValueError("Field inputs or plan settings changed since planning")
    root = _in_shared_root(Path(output).expanduser().resolve(), profile)
    settings = read_json(value["config"]["path"])
    settings["Problem"]["Output"] = str(root / "raw")
    settings["Model"]["Mesh"] = value["mesh"]
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "plan.json", value)
    atomic_json(root / "palace.json", settings)
    directives = [
        "#!/bin/bash", "#SBATCH --job-name=" + profile.get("job_name", "emulateqpu-field"),
        f"#SBATCH --nodes={profile['nodes']}", f"#SBATCH --ntasks={profile['mpi_ranks']}",
        "#SBATCH --cpus-per-task=1", f"#SBATCH --mem={profile['memory_mb']}M",
        f"#SBATCH --time={profile['wall_time']}",
        "#SBATCH --output=" + hpc._directive(root / "slurm-%j.out"),
        "#SBATCH --chdir=" + hpc._directive(Path(value["config"]["path"]).parent),
    ]
    for field in ("partition", "account", "qos"):
        if field in profile:
            directives.append(f"#SBATCH --{field}={profile[field]}")
    worker = [profile["python"], "-m", "emulateqpu.field_slurm", str(root)]
    script = root / "job.sbatch"
    script.write_text("\n".join([*directives, "", "set -euo pipefail",
                                  "exec " + shlex.join(worker), ""]), encoding="utf-8", newline="\n")
    script.chmod(0o700)
    record = {"schema": RUN_SCHEMA, "created_utc": _now(), "plan_sha256": file_hash(root / "plan.json"),
              "config_sha256": file_hash(root / "palace.json"), "script_sha256": file_hash(script),
              "command": _command(profile, root / "palace.json"), "status": "RENDERED"}
    atomic_json(root / "run.json", record)
    return record


def _run(root):
    root = Path(root).expanduser().resolve()
    if (root / "run.json").is_symlink():
        raise ValueError("The field run manifest cannot be a symlink")
    record = read_json(root / "run.json")
    if record.get("schema") != RUN_SCHEMA:
        raise ValueError("Expected a rendered Palace Slurm field run")
    for path, key in (("plan.json", "plan_sha256"), ("palace.json", "config_sha256"),
                      ("job.sbatch", "script_sha256")):
        if (root / path).is_symlink() or file_hash(root / path) != record[key]:
            raise ValueError("Rendered field run changed: " + path)
    value = read_json(root / "plan.json")
    if record["command"] != _command(value["profile"], root / "palace.json"):
        raise ValueError("Rendered Palace command changed or was moved")
    return root, record, value


def submit(root, *, runner=None):
    """Submit once, preserving indeterminate submissions for manual inspection."""
    root, record, value = _run(root)
    _verify_inputs(value)
    if (root / "raw").exists() or (root / "execution.json").exists():
        raise ValueError("Use a fresh rendered directory for each field submission")
    lock = root / "submission.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        receipt = root / "submission.json"
        if receipt.exists():
            raise ValueError("A field submission already exists; create a new rendered directory for a retry")
        command = ["sbatch", "--parsable", str(root / "job.sbatch")]
        state = {"schema": SUBMISSION_SCHEMA, "command": command,
                 "plan_sha256": record["plan_sha256"], "submitted_utc": _now()}
        try:
            response = (runner or subprocess.run)(command, capture_output=True, text=True, timeout=30, check=False)
            state.update(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr)
            if response.returncode == 0:
                state.update(hpc.parse_submission(response.stdout), status="SUBMITTED")
            else:
                state.update(status="SUBMISSION_FAILED")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            state.update(status="SUBMISSION_UNKNOWN", error=str(error))
        atomic_json(receipt, state)
        if state["status"] != "SUBMITTED":
            raise RuntimeError("Field job identity was not established; inspect submission.json before retrying")
        return state
    finally:
        lock.unlink()


def _check_csv(path, minimum):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required field output table is missing or invalid: {path.name}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        headers = [cell.strip() for cell in next(reader, ())]
        if not headers or not all(headers) or len(headers) != len(set(headers)):
            raise ValueError(f"Invalid output table headers: {path.name}")
        if path.name == "eig.csv" and not {"m", "Re{f} (GHz)"}.issubset(headers):
            raise ValueError("Palace eigenmode table lacks mode and frequency columns")
        if path.name == "port-S.csv" and ("f (GHz)" not in headers
                                           or not any(h.startswith("|S[") and h.endswith("(dB)") for h in headers)):
            raise ValueError("Palace scattering table lacks frequency and magnitude columns")
        if path.name == "terminal-C.csv" and ("i" not in headers
                                               or sum(h.startswith("C[i][") and h.endswith("(F)") for h in headers) < minimum):
            raise ValueError("Palace terminal capacitance table lacks the expected terminal columns")
        count = 0
        for line in reader:
            if not line:
                continue
            if len(line) != len(headers):
                raise ValueError(f"Invalid output table width: {path.name}")
            for header, value in zip(headers, line, strict=True):
                number = float(value)
                allowed_infinity = (header.startswith("Q") and number == math.inf) or (
                    header.endswith("(dB)") and number == -math.inf)
                if not math.isfinite(number) and not allowed_infinity:
                    raise ValueError(f"Invalid numeric value in {path.name}: {header}")
            count += 1
        if count < minimum:
            raise ValueError(f"{path.name} has {count} rows, expected at least {minimum}")


def _outputs(raw, required, mpi_ranks):
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError("Palace did not produce the expected raw output directory")
    metadata_path = raw / "palace.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("Palace did not produce native run metadata in palace.json")
    metadata = read_json(metadata_path)
    problem = metadata.get("Problem") if isinstance(metadata, dict) else None
    actual = problem.get("MPISize") if isinstance(problem, dict) else None
    if type(actual) is not int or actual != mpi_ranks:
        raise ValueError(f"Palace reported MPI size {actual}, expected {mpi_ranks}")
    for name, minimum in required.items():
        _check_csv(raw / name, minimum)
    return {"actual_mpi_ranks": actual, "files": _file_manifest(raw)}


def _palace_log_hashes(directory):
    paths = [directory / name for name in PALACE_LOG_NAMES]
    if directory.is_symlink() or any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Missing or unsafe captured Palace log")
    return {path.name: file_hash(path) for path in paths}


def _execute(root, *, runner=None, environment=None):
    root, record, value = _run(root)
    _verify_inputs(value)
    profile = value["profile"]
    environment = dict(os.environ if environment is None else environment)
    job = environment.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"\d+", job) or environment.get("SLURM_NTASKS") != str(profile["mpi_ranks"]):
        raise ValueError("Palace batch worker requires the planned Slurm job allocation")
    if (root / "raw").exists() or (root / "execution.json").exists():
        raise ValueError("The field run already has outputs")
    if any(int(environment.get(name, "1")) > 1 for name in ("PMI_SIZE", "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE")):
        raise ValueError("Palace field worker must start in the batch shell, outside any MPI rank")
    with (root / "execution.lock").open("x"):
        pass
    result = {"schema": EXECUTION_SCHEMA, "plan_sha256": record["plan_sha256"], "job_id": job,
              "requested_mpi_ranks": profile["mpi_ranks"], "command": record["command"],
              "field_numerical_qualification": "NOT_RUN", "status": "FAILED"}
    started = time.monotonic()
    worker_env = {**environment, **profile.get("environment", {}), **dict.fromkeys(THREAD_VARIABLES, "1")}
    try:
        logs = root / PALACE_LOG_DIRECTORY
        logs.mkdir()
        with (logs / PALACE_LOG_NAMES[0]).open("xb") as stdout, (logs / PALACE_LOG_NAMES[1]).open("xb") as stderr:
            response = (runner or subprocess.run)(record["command"], cwd=Path(value["config"]["path"]).parent,
                                                   env=worker_env, check=False, stdout=stdout, stderr=stderr)
        result["logs_sha256"] = _palace_log_hashes(logs)
        result["returncode"] = response.returncode
        if response.returncode == 0:
            result.update(_outputs(root / "raw", value["required_outputs"], profile["mpi_ranks"]), status="COMPLETE")
    except (OSError, ValueError, csv.Error, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    result["elapsed_s"] = time.monotonic() - started
    atomic_json(root / "execution.json", result)
    return result


def collect(root):
    """Verify field output bytes locally. Scheduler and numerical acceptance are separate."""
    root, record, value = _run(root)
    result = {"schema": COLLECTION_SCHEMA, "status": "INCOMPLETE", "complete": False, "plan_sha256": record["plan_sha256"],
              "raw_directory": str(root / "raw"), "field_numerical_qualification": "NOT_RUN"}
    if not (root / "execution.json").is_file():
        return result
    try:
        if (root / "execution.json").is_symlink():
            raise ValueError("The field execution receipt cannot be a symlink")
        execution = read_json(root / "execution.json")
        if (execution.get("schema") != EXECUTION_SCHEMA or execution["plan_sha256"] != record["plan_sha256"]
                or execution["command"] != record["command"]):
            raise ValueError("Execution is not bound to the rendered field plan")
        result["execution"] = execution
        if execution["status"] == "COMPLETE":
            outputs = _outputs(root / "raw", value["required_outputs"], value["profile"]["mpi_ranks"])
            if execution["returncode"] != 0 or any(execution[key] != observed for key, observed in outputs.items()):
                raise ValueError("Field output contents changed after execution")
            if ("logs_sha256" in execution
                    and execution["logs_sha256"] != _palace_log_hashes(root / PALACE_LOG_DIRECTORY)):
                raise ValueError("Captured Palace logs changed after execution")
            result.update(status="COMPLETE", complete=True)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        result.update(status="CORRUPT", error=str(error))
    return result


def _scheduler(job, cluster, runner):
    suffix = ["--clusters=" + cluster] if cluster else []
    jobs = ["--jobs=" + job, "--noheader"]
    commands = [(["squeue", *jobs, "--format=%i|%T", *suffix], "queue"),
                (["sacct", *jobs, "--parsable2",
                  "--format=JobIDRaw%64,State%32,ExitCode,NTasks,ElapsedRaw,MaxRSS", *suffix], "accounting")]
    rows = {}
    errors = []
    for command, source in commands:
        try:
            response = runner(command, capture_output=True, text=True, timeout=15, check=False)
            if response.returncode != 0:
                errors.append({"source": source, "error": response.stderr.strip()})
                continue
            for line in response.stdout.splitlines():
                fields = [cell.strip() for cell in line.split("|")]
                if len(fields) < 2 or not (fields[0] == job or fields[0].startswith(job + ".")):
                    continue
                row = {"id": fields[0], "state": hpc._state(fields[1]), "source": source}
                if source == "accounting" and len(fields) >= 6:
                    row.update(exit_code=fields[2], tasks=int(fields[3]) if fields[3].isdigit() else None,
                               elapsed_seconds=int(fields[4]) if fields[4].isdigit() else None,
                               max_task_rss=fields[5] or None)
                rows.setdefault(fields[0], row)
        except (OSError, subprocess.SubprocessError) as error:
            errors.append({"source": source, "error": str(error)})
    return rows, errors


def status(root, *, runner=None):
    root, record, value = _run(root)
    collection = collect(root)
    identity = {"schema": STATUS_SCHEMA, "plan_sha256": record["plan_sha256"], "result": collection}
    receipt = root / "submission.json"
    if receipt.is_symlink():
        raise ValueError("The field submission receipt cannot be a symlink")
    if not receipt.exists():
        return {**identity, "status": "RENDERED", "complete": False}
    submission = read_json(receipt)
    if submission["schema"] != SUBMISSION_SCHEMA or submission["plan_sha256"] != record["plan_sha256"]:
        raise ValueError("Submission is not bound to the rendered field plan")
    job = submission.get("job_id")
    if not job:
        return {**identity, "status": submission["status"], "complete": False}
    if {"job_id": job, "cluster": submission.get("cluster")} != hpc.parse_submission(submission["stdout"]):
        raise ValueError("The stored Slurm job identity differs from the scheduler receipt")
    rows, errors = _scheduler(job, submission.get("cluster"), runner or subprocess.run)
    scheduler = rows.get(job, {})
    steps = [row for name, row in rows.items() if re.fullmatch(re.escape(job) + r"\.\d+", name)]
    expected = value["profile"]["mpi_ranks"]
    ranks_confirmed = (len(steps) == 1 and steps[0]["state"] == "COMPLETED"
                       and steps[0].get("exit_code") == "0:0" and steps[0].get("tasks") == expected)
    completed = (scheduler.get("state") == "COMPLETED" and scheduler.get("exit_code") == "0:0"
                 and ranks_confirmed and collection["complete"] and collection["execution"]["job_id"] == job)
    return {**identity, "status": "COMPLETE" if completed else "INCOMPLETE", "complete": completed,
            "scheduler_state": scheduler.get("state", "UNKNOWN"), "requested_mpi_ranks": expected,
            "ranks_confirmed": ranks_confirmed, "scheduler_job": scheduler, "scheduler_steps": steps,
            "scheduler_errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Internal worker for a rendered Palace Slurm job")
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    return 0 if _execute(args.run)["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
