"""Explicit Slurm job-array adapter; no scheduler action at import or rendering.

This adapter distributes independent immutable requests. It does not distribute
one SciPy state vector across nodes. Verification in this release uses mocks;
site acceptance is a separate execution record.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from .execution import (_load_run, _now, _read, atomic_json, collect_run,
                        normalize_study, prepare_run, THREAD_VARIABLES)

PROFILE_SCHEMA = "emulateqpu.slurm_profile.v1"
OFFICIAL_SOURCES = {
    "access_date": "2026-11-16",
    "sbatch": "https://slurm.schedmd.com/sbatch.html",
    "arrays": "https://slurm.schedmd.com/job_array.html",
    "accounting": "https://slurm.schedmd.com/sacct.html",
}


def _load(value):
    return _read(value) if isinstance(value, (str, Path)) else deepcopy(value)


def wall_seconds(value):
    if not isinstance(value, str):
        raise ValueError("wall_time must use [days-]hours:minutes:seconds")
    match = re.fullmatch(r"(?:(\d+)-)?(\d{1,3}):(\d{2}):(\d{2})", value)
    if not match:
        raise ValueError("wall_time must use [days-]hours:minutes:seconds")
    day, hour, minute, second = (int(x or 0) for x in match.groups())
    if minute >= 60 or second >= 60 or (day and hour >= 24):
        raise ValueError("Invalid Slurm wall_time")
    total = 86400 * day + 3600 * hour + 60 * minute + second
    if total <= 0:
        raise ValueError("Positive wall_time required")
    return total


def validate_profile(value):
    profile = _load(value)
    if not isinstance(profile, dict):
        raise ValueError("A site-profile object is required")
    allowed = {"schema", "partition", "account", "qos", "memory_mb", "wall_time",
               "cpus_per_task", "array_concurrency", "environment", "python",
               "shared_filesystem", "shared_root", "dependency", "job_name", "max_retries"}
    if set(profile) - allowed:
        raise ValueError("Unknown site profile fields: " + str(sorted(set(profile) - allowed)))
    if profile.get("schema", PROFILE_SCHEMA) != PROFILE_SCHEMA:
        raise ValueError("Unsupported site-profile schema")
    if profile.get("shared_filesystem") is not True:
        raise ValueError("This adapter requires an explicitly shared filesystem; staging is not implemented")
    for name, default, maximum in (("memory_mb", 1024, 1048576), ("cpus_per_task", 1, 256),
                                   ("array_concurrency", 1, 64), ("max_retries", 0, 3)):
        number = profile.get(name, default)
        if type(number) is not int or not (0 if name == "max_retries" else 1) <= number <= maximum:
            raise ValueError("Invalid " + name)
        profile[name] = number
    profile["wall_time"] = profile.get("wall_time", "00:05:00")
    wall_seconds(profile["wall_time"])
    profile["python"] = profile.get("python", sys.executable)
    if not isinstance(profile["python"], str) or not Path(profile["python"]).is_absolute():
        raise ValueError("python must identify the absolute shared installed interpreter")
    for name in ("partition", "account", "qos", "job_name"):
        if name in profile and (not isinstance(profile[name], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", profile[name])):
            raise ValueError("Invalid scheduler " + name)
    if "dependency" in profile and not re.fullmatch(r"(?:afterok|afterany|afternotok):\d+(?::\d+)*", profile["dependency"]):
        raise ValueError("dependency must be an explicit afterok/afterany/afternotok job list")
    environment = profile.setdefault("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("environment must be a string mapping")
    for key, text in environment.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or not isinstance(text, str) or "\n" in text or "\x00" in text:
            raise ValueError("Invalid environment entry")
        if key in THREAD_VARIABLES and text != "1":
            raise ValueError("Each quantum task uses exactly one numerical thread")
    if "shared_root" in profile:
        if not isinstance(profile["shared_root"], str) or not Path(profile["shared_root"]).is_absolute():
            raise ValueError("shared_root must be absolute")
    profile["schema"] = PROFILE_SCHEMA
    return profile


def _directive(value):
    text = str(value)
    if any(c in text for c in ("\n", "\r", "\x00", "$", "`")):
        raise ValueError("Scheduler directives require literal values, without shell expansion")
    return json.dumps(text) if any(c.isspace() or c == '"' for c in text) else text


def render(plan_or_study, profile, output):
    """Render a shared-filesystem run and script without submitting anything."""
    profile = validate_profile(profile)
    value = _load(plan_or_study)
    if isinstance(value, dict) and value.get("schema") == "emulateqpu.execution_plan.v1":
        value = value["study"]
    elif isinstance(value, dict) and "request" in value and value.get("schema") == "emulateqpu.plan.v1":
        value = value["request"]
    study = normalize_study(value)
    root = Path(output).expanduser().resolve()
    if "shared_root" in profile and not root.is_relative_to(Path(profile["shared_root"]).resolve()):
        raise ValueError("Rendered run must reside beneath the declared shared_root")
    for task in study["tasks"]:
        resources = task["request"].get("resources", {})
        if resources.get("memory_mb", 1024) > profile["memory_mb"]:
            raise ValueError("Site memory allocation is smaller than a task's declared memory budget")
        if resources.get("wall_time_s", 120) > wall_seconds(profile["wall_time"]):
            raise ValueError("Site wall allocation is smaller than a task's declared wall budget")
    root = prepare_run(study, root, backend="slurm", workers=profile["array_concurrency"],
                       memory_limit_bytes=profile["memory_mb"] * 1024**2 * profile["array_concurrency"],
                       max_retries=profile["max_retries"])
    (root / "logs").mkdir()
    directives = ["#!/bin/bash", "#SBATCH --job-name=" + profile.get("job_name", "emulateqpu"),
                  "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
                  "#SBATCH --cpus-per-task=" + str(profile["cpus_per_task"]),
                  "#SBATCH --mem=" + str(profile["memory_mb"]) + "M",
                  "#SBATCH --time=" + profile["wall_time"],
                  "#SBATCH --array=0-%d%%%d" % (len(study["tasks"]) - 1, profile["array_concurrency"]),
                  "#SBATCH --output=" + _directive(str(root / "logs" / "%A_%a.out")),
                  "#SBATCH --error=" + _directive(str(root / "logs" / "%A_%a.err")),
                  "#SBATCH --chdir=" + _directive(str(root))]
    for name in ("partition", "account", "qos", "dependency"):
        if name in profile:
            directives.append("#SBATCH --" + name + "=" + _directive(profile[name]))
    lines = directives + ["", "set -euo pipefail"]
    environment = {**profile["environment"], **{key: "1" for key in THREAD_VARIABLES}}
    for key, text in sorted(environment.items()):
        lines.append("export " + key + "=" + shlex.quote(text))
    lines += [': "${SLURM_ARRAY_TASK_ID:?Slurm array task id required}"',
              "exec " + shlex.quote(profile["python"]) + " -m emulateqpu.execution --run-task " +
              shlex.quote(str(root)) + ' "$SLURM_ARRAY_TASK_ID"', ""]
    script = root / "job.sbatch"
    script.write_text("\n".join(lines))
    script.chmod(0o700)
    metadata = {"schema": "emulateqpu.slurm_render.v1", "created_utc": _now(),
                "profile": profile, "script": "job.sbatch",
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "study_sha256": _read(root / "run.json")["study_sha256"],
                "task_map": _read(root / "run.json")["tasks"],
                "rendered": True, "cluster_executed": False,
                "documentation": OFFICIAL_SOURCES}
    atomic_json(root / "slurm.json", metadata)
    return metadata


def parse_submission(text):
    match = re.fullmatch(r"\s*(\d+)(?:;([A-Za-z0-9_.-]+))?\s*", text)
    if not match:
        raise ValueError("sbatch --parsable returned an invalid job identity")
    return {"job_id": match.group(1), "cluster": match.group(2)}


def _rendered(root):
    root, manifest, study, plan = _load_run(root)
    metadata = _read(root / "slurm.json")
    if metadata.get("schema") != "emulateqpu.slurm_render.v1" or metadata.get("study_sha256") != manifest["study_sha256"]:
        raise ValueError("Rendered Slurm identity is incompatible")
    if metadata.get("task_map") != manifest["tasks"]:
        raise ValueError("Rendered array mapping changed")
    if hashlib.sha256((root / metadata["script"]).read_bytes()).hexdigest() != metadata["script_sha256"]:
        raise ValueError("Rendered scheduler script changed")
    return root, manifest, metadata


def submit(root, *, retry=False, runner=None):
    """Explicit external action. Calling render or status never submits a job."""
    root, _, _ = _rendered(root)
    lock = root / "submission.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("Submission already in progress or indeterminate; inspect the preserved lock") from error
    os.write(fd, (json.dumps({"controller_pid": os.getpid(), "created_utc": _now()}) + "\n").encode())
    os.fsync(fd); os.close(fd)
    try:
        return _submit_locked(root, retry=retry, runner=runner)
    finally:
        lock.unlink(missing_ok=True)


def _submit_locked(root, *, retry=False, runner=None):
    runner = runner or subprocess.run
    root, manifest, metadata = _rendered(root)
    previous = sorted(root.glob("submission_*.json"))
    if previous and not retry:
        raise ValueError("A submission exists; duplicate submission rejected")
    if len(previous) > manifest["max_retries"]:
        raise ValueError("Slurm retry allowance exhausted")
    command = ["sbatch", "--parsable"]
    if previous:
        # An explicit retry must still establish that prior tasks have stopped.
        prior_status = status(root, runner=runner)
        if any(t["scheduler_state"] in ("UNKNOWN", "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED")
               for t in prior_status["tasks"] if t["result_status"] != "COMPLETE"):
            raise ValueError("Prior task termination is not established; refusing concurrent retry")
        collection = collect_run(root)
        if any(t["status"] in ("CORRUPT", "DUPLICATE") for t in collection["tasks"]):
            raise ValueError("Corrupt/duplicate results cannot be retried")
        pending = [str(t["index"]) for t in collection["tasks"] if t["status"] != "COMPLETE"]
        if not pending:
            raise ValueError("All tasks already complete")
        command.append("--array=" + ",".join(pending) + "%" + str(metadata["profile"]["array_concurrency"]))
    command.append(str(root / metadata["script"]))
    try:
        completed = runner(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        completed = subprocess.CompletedProcess(command, 127, "", str(error))
    record = {"schema": "emulateqpu.slurm_submission.v1", "created_utc": _now(),
              "command": command, "returncode": completed.returncode,
              "stdout": completed.stdout, "stderr": completed.stderr,
              "study_sha256": manifest["study_sha256"], "attempt": len(previous) + 1,
              "cluster_executed": False}
    if completed.returncode == 0:
        try:
            record.update(parse_submission(completed.stdout), status="SUBMITTED")
        except ValueError as error:
            record.update(status="SUBMISSION_ID_UNKNOWN", error=str(error))
    else:
        record["status"] = "SUBMISSION_FAILED"
    atomic_json(root / ("submission_%04d.json" % (len(previous) + 1)), record)
    if record["status"] != "SUBMITTED":
        raise RuntimeError("Slurm submission did not establish a job id; inspect the preserved record before retry")
    return record


def _state(value):
    return value.strip().split()[0].split("+")[0].upper() if value.strip() else "UNKNOWN"


def status(root, *, runner=None):
    runner = runner or subprocess.run
    root, manifest, metadata = _rendered(root)
    collection = collect_run(root, write=False)
    submissions = sorted(root.glob("submission_*.json"))
    states = {}
    errors = []
    latest = _read(submissions[-1]) if submissions else None
    if latest and latest.get("job_id"):
        job = latest["job_id"]
        cluster = ["--clusters=" + latest["cluster"]] if latest.get("cluster") else []
        commands = [(["squeue", "--array", "--noheader", "--jobs=" + job, "--format=%i|%T"] + cluster, "queue"),
                    (["sacct", "--array", "--noheader", "--parsable2", "--jobs=" + job, "--format=JobID%64,State%32,ExitCode"] + cluster, "accounting")]
        for command, source in commands:
            try:
                response = runner(command, capture_output=True, text=True, timeout=15, check=False)
                if response.returncode:
                    errors.append({"source": source, "error": response.stderr.strip()})
                    continue
                for line in response.stdout.splitlines():
                    parts = line.strip().split("|")
                    if len(parts) < 2:
                        continue
                    match = re.fullmatch(re.escape(job) + r"_(\d+)", parts[0])
                    if match:
                        index = int(match.group(1))
                        current = _state(parts[1])
                        if source == "accounting" and current == "COMPLETED" and len(parts) > 2 and parts[2] != "0:0":
                            current = "FAILED"
                        # Active queue states outrank stale accounting rows.
                        if index not in states:
                            states[index] = {"state": current, "source": source}
            except (OSError, subprocess.SubprocessError) as error:
                errors.append({"source": source, "error": str(error)})
    rows = []
    for row in collection["tasks"]:
        scheduler = states.get(row["index"], {"state": "UNKNOWN", "source": None})
        state = scheduler["state"]
        effective = row["status"]
        if effective != "COMPLETE":
            if state == "COMPLETED":
                effective = "MISSING_RESULT" if row["status"] == "MISSING" else row["status"]
            elif state in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"):
                effective = "SCHEDULER_FAILED"
            elif not submissions:
                effective = "RENDERED"
            elif effective == "MISSING":
                effective = state
        rows.append({"id": row["id"], "index": row["index"], "scheduler_state": state,
                     "scheduler_source": scheduler["source"], "result_status": row["status"], "status": effective})
    return {"schema": "emulateqpu.slurm_status.v1", "checked_utc": _now(),
            "status": "COMPLETE" if collection["complete"] else ("RENDERED" if not submissions else "INCOMPLETE"),
            "complete": collection["complete"], "submission": latest, "tasks": rows,
            "scheduler_errors": errors, "result_collection": collection,
            "cluster_acceptance_qualified": False}


def collect(root):
    return collect_run(root)
