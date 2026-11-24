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
RENDER_SCHEMA = "emulateqpu.slurm_render.v1"
SUBMISSION_SCHEMA = "emulateqpu.slurm_submission.v1"
DEFAULT_MAX_ARRAY_SIZE = 1001
TERMINAL_STATES = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
                   "NODE_FAIL", "OUT_OF_MEMORY", "TIMEOUT"}
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
               "cpus_per_task", "array_concurrency", "max_array_size", "environment", "python",
               "shared_filesystem", "shared_root", "dependency", "job_name", "max_retries"}
    if set(profile) - allowed:
        raise ValueError("Unknown site profile fields: " + str(sorted(set(profile) - allowed)))
    if profile.get("schema", PROFILE_SCHEMA) != PROFILE_SCHEMA:
        raise ValueError("Unsupported site-profile schema")
    if profile.get("shared_filesystem") is not True:
        raise ValueError("This adapter requires an explicitly shared filesystem; staging is not implemented")
    for name, default, maximum in (("memory_mb", 1024, 1048576), ("cpus_per_task", 1, 256),
                                   ("array_concurrency", 1, 64), ("max_retries", 0, 3),
                                   ("max_array_size", DEFAULT_MAX_ARRAY_SIZE, 10000)):
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
        if key.startswith(("SLURM_", "SBATCH_", "PMI_", "PMIX_", "OMPI_COMM_WORLD_")):
            raise ValueError("Profiles cannot override Slurm or MPI allocation variables")
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
    shards = []
    max_size = profile["max_array_size"]
    for start in range(0, len(study["tasks"]), max_size):
        stop = min(start + max_size, len(study["tasks"]))
        script = "job.sbatch" if start == 0 else f"job-{len(shards) + 1:04d}.sbatch"
        script_sha256 = _write_batch_script(root, profile, script, start, stop)
        shards.append({"start": start, "stop": stop, "script": script, "script_sha256": script_sha256})
    manifest = _read(root / "run.json")
    metadata = {"schema": RENDER_SCHEMA, "created_utc": _now(), "profile": profile,
                "script": shards[0]["script"], "script_sha256": shards[0]["script_sha256"],
                "shards": shards, "study_sha256": manifest["study_sha256"], "task_map": manifest["tasks"],
                "rendered": True, "cluster_executed": False, "documentation": OFFICIAL_SOURCES}
    atomic_json(root / "slurm.json", metadata)
    return metadata


def _write_batch_script(root, profile, name, start, stop):
    size = stop - start
    directives = ["#!/bin/bash", "#SBATCH --job-name=" + profile.get("job_name", "emulateqpu"),
                  "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --no-requeue",
                  "#SBATCH --cpus-per-task=" + str(profile["cpus_per_task"]),
                  "#SBATCH --mem=" + str(profile["memory_mb"]) + "M",
                  "#SBATCH --time=" + profile["wall_time"],
                  f"#SBATCH --array=0-{size - 1}%{profile['array_concurrency']}",
                  "#SBATCH --output=" + _directive(str(root / "logs" / "%A_%a.out")),
                  "#SBATCH --error=" + _directive(str(root / "logs" / "%A_%a.err")),
                  "#SBATCH --chdir=" + _directive(str(root))]
    for key in ("partition", "account", "qos", "dependency"):
        if key in profile:
            directives.append("#SBATCH --" + key + "=" + _directive(profile[key]))
    lines = [*directives, "", "set -euo pipefail"]
    environment = {**profile["environment"], **dict.fromkeys(THREAD_VARIABLES, "1")}
    for key, value in sorted(environment.items()):
        lines.append("export " + key + "=" + shlex.quote(value))
    array_index = '"$SLURM_ARRAY_TASK_ID"' if start == 0 else f'"$((SLURM_ARRAY_TASK_ID + {start}))"'
    lines.extend([
        ': "${SLURM_ARRAY_TASK_ID:?Slurm array task id required}"',
        f'[[ "$SLURM_ARRAY_TASK_ID" =~ ^(0|[1-9][0-9]*)$ && "$SLURM_ARRAY_TASK_ID" -lt {size} ]]',
        "exec " + shlex.join([profile["python"], "-m", "emulateqpu.execution", "--run-task", str(root)])
        + " " + array_index, "",
    ])
    script = root / name
    script.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    script.chmod(0o700)
    return hashlib.sha256(script.read_bytes()).hexdigest()


def parse_submission(text):
    match = re.fullmatch(r"\s*(\d+)(?:;([A-Za-z0-9_.-]+))?\s*", text)
    if not match:
        raise ValueError("sbatch --parsable returned an invalid job identity")
    return {"job_id": match.group(1), "cluster": match.group(2)}


def _rendered(root):
    root, manifest, _, _ = _load_run(root)
    metadata = _read(root / "slurm.json")
    if metadata.get("schema") != RENDER_SCHEMA or metadata.get("study_sha256") != manifest["study_sha256"]:
        raise ValueError("Rendered Slurm identity is incompatible")
    if metadata.get("task_map") != manifest["tasks"]:
        raise ValueError("Rendered array mapping changed")
    # Older single-array renders used only the top-level script identity.
    scripts = {key: metadata[key] for key in ("script", "script_sha256")}
    shards = metadata.get("shards", [{"start": 0, "stop": len(manifest["tasks"]), **scripts}])
    end = 0
    for number, shard in enumerate(shards, 1):
        expected = "job.sbatch" if number == 1 else f"job-{number:04d}.sbatch"
        if (shard["start"] != end or type(shard["stop"]) is not int
                or not end < shard["stop"] <= len(manifest["tasks"]) or shard["script"] != expected):
            raise ValueError("Rendered shard mapping changed")
        script = root / expected
        if script.is_symlink() or hashlib.sha256(script.read_bytes()).hexdigest() != shard["script_sha256"]:
            raise ValueError("Rendered scheduler script changed")
        end = shard["stop"]
    if end != len(manifest["tasks"]) or any(shards[0][key] != value for key, value in scripts.items()):
        raise ValueError("Rendered shard mapping changed")
    return root, manifest, {**metadata, "shards": shards}


def _submission_path(root, attempt):
    return root / f"submission_{attempt:04d}.json"


def _history(root, manifest, metadata):
    history = []
    paths = sorted(root.glob("submission_*.json"), key=lambda path: (len(path.name), path.name))
    for attempt, path in enumerate(paths, 1):
        if path != _submission_path(root, attempt) or path.is_symlink():
            raise ValueError("Slurm submission receipt identity changed")
        receipt = _read(path)
        if (receipt.get("schema") != SUBMISSION_SCHEMA or receipt.get("attempt") != attempt
                or receipt.get("study_sha256") != manifest["study_sha256"]):
            raise ValueError("Slurm submission receipt identity changed")
        number = receipt.get("shard", 1)
        if type(number) is not int or not 1 <= number <= len(metadata["shards"]):
            raise ValueError("Slurm submission shard changed")
        shard = metadata["shards"][number - 1]
        indices = _submission_indices(receipt["command"], shard)
        if receipt.get("task_indices", indices) != indices:
            raise ValueError("Slurm submission task mapping changed")
        if receipt.get("status") == "SUBMITTED" and parse_submission(receipt["stdout"]) != {
                "job_id": receipt.get("job_id"), "cluster": receipt.get("cluster")}:
            raise ValueError("Slurm submission job identity changed")
        history.append({**receipt, "shard": number, "task_indices": indices})
    return history


def _submission_indices(command, shard):
    if not command or Path(command[-1]).name != shard["script"]:
        raise ValueError("Slurm submission script identity changed")
    start, stop = shard["start"], shard["stop"]
    overrides = [arg for arg in command if arg.startswith("--array=")]
    if not overrides:
        return list(range(start, stop))
    match = re.fullmatch(r"--array=(\d+(?:-\d+|(?:,\d+)*))%\d+", overrides[0]) if len(overrides) == 1 else None
    if not match:
        raise ValueError("Slurm submission array expression changed")
    selector = match[1]
    if "-" in selector:
        lo, hi = map(int, selector.split("-"))
        if not 0 <= lo <= hi < stop - start:
            raise ValueError("Slurm submission task mapping changed")
        local = range(lo, hi + 1)
    else:
        local = [int(index) for index in selector.split(",")]
        if local != sorted(set(local)) or local[-1] >= stop - start:
            raise ValueError("Slurm submission task mapping changed")
    return [start + index for index in local]


def submit(root, *, retry=False, runner=None):
    """Explicit external action. Calling render or status never submits a job."""
    root, _, _ = _rendered(root)
    lock = root / "submission.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("Submission already in progress or indeterminate; inspect the preserved lock") from error
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"controller_pid": os.getpid(), "created_utc": _now()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return _submit_locked(root, retry=retry, runner=runner or subprocess.run)
    finally:
        lock.unlink(missing_ok=True)


def _retry_pending(root, manifest, metadata, history, runner):
    collection = collect_run(root, write=False)
    if collection["unexpected_task_directories"] or any(t["status"] in ("CORRUPT", "DUPLICATE") for t in collection["tasks"]):
        raise ValueError("Corrupt/duplicate results cannot be retried")
    pending = [t for t in collection["tasks"] if t["status"] != "COMPLETE"]
    if not pending:
        raise ValueError("All tasks already complete")
    attempts = {t["index"]: 0 for t in pending}
    for receipt in history:
        for index in receipt["task_indices"]:
            if index in attempts:
                attempts[index] += 1
    if any(count > manifest["max_retries"] for count in attempts.values()):
        raise ValueError("Slurm retry allowance exhausted for at least one pending task")
    states, _ = _scheduler_states(history, metadata, runner, set(attempts))
    archives = []
    for task in pending:
        index = task["index"]
        scheduler = states.get(index, {})
        if attempts[index] and not scheduler.get("terminal", False):
            raise ValueError(f"Prior task termination is not established for {task['id']}; refusing concurrent retry")
        lock = root / "tasks" / task["id"] / "dispatch.lock"
        if lock.exists() or lock.is_symlink():
            archives.append((lock, _dispatch_archive(lock, manifest["tasks"][index], scheduler)))
    # Check every lock first. Archive only known stopped dispatches, leaving their
    # original bytes and failed attempts available for inspection and resume.
    for lock, archive in archives:
        lock.rename(archive)
    return set(attempts)


def _dispatch_archive(lock, identity, scheduler):
    message = (f"Dispatch lock remains for {identity['id']}. Verify the prior controller and worker have stopped "
               "and archive the lock before retrying")
    owner = scheduler.get("owner") if scheduler.get("terminal") else None
    if not owner or lock.is_symlink() or not lock.is_file():
        raise ValueError(message)
    try:
        record = _read(lock)
    except (OSError, ValueError) as error:
        raise ValueError(message) from error
    if (not isinstance(record, dict) or record.get("task_id") != identity["id"]
            or record.get("request_sha256") != identity["request_sha256"] or record.get("slurm") != owner):
        raise ValueError(message)
    archive = lock.with_name(f"dispatch-lock-{owner['cluster']}-{owner['job_id']}.json")
    if archive.exists() or archive.is_symlink():
        raise ValueError(message)
    return archive


def _submit_locked(root, *, retry, runner):
    root, manifest, metadata = _rendered(root)
    history = _history(root, manifest, metadata)
    if history and not retry:
        raise ValueError("A submission exists; duplicate submission rejected")
    if any(receipt["status"] != "SUBMITTED" for receipt in history):
        raise ValueError("Prior Slurm submission has no confirmed job identity; inspect the preserved record")
    pending = (_retry_pending(root, manifest, metadata, history, runner) if history
               else set(range(len(manifest["tasks"]))))
    profile = metadata["profile"]
    records = []
    for number, shard in enumerate(metadata["shards"], 1):
        indices = sorted(pending.intersection(range(shard["start"], shard["stop"])))
        if not indices:
            continue
        command = ["sbatch", "--parsable", "--no-requeue"]
        dependency = profile.get("dependency")
        if records:
            # Serialize shards to preserve the study-wide concurrency cap.
            previous = records[-1]
            after_previous = "afterany:" + previous["job_id"]
            dependency = f"{after_previous},{dependency}" if dependency else after_previous
            if previous.get("cluster"):
                command.append("--clusters=" + previous["cluster"])
        if dependency:
            command.append("--dependency=" + dependency)
        local = (",".join(str(index - shard["start"]) for index in indices) if history
                 else f"0-{shard['stop'] - shard['start'] - 1}")
        command.append(f"--array={local}%{profile['array_concurrency']}")
        command.append(str(root / shard["script"]))
        attempt = len(history) + len(records) + 1
        receipt = {"schema": SUBMISSION_SCHEMA, "created_utc": _now(), "command": command,
                   "study_sha256": manifest["study_sha256"], "attempt": attempt, "shard": number,
                   "task_indices": indices, "cluster_executed": False}
        records.append(_submit_shard(_submission_path(root, attempt), receipt, runner))
    if len(records) == 1:
        return records[0]
    return {"status": "SUBMITTED", "submissions": records, "shard_count": len(records),
            "study_sha256": manifest["study_sha256"], "cluster_executed": False}


def _submit_shard(path, record, runner):
    command = record["command"]
    atomic_json(path, {**record, "status": "SUBMITTING"})
    try:
        completed = runner(command, capture_output=True, text=True, timeout=30, check=False)
        record.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        if completed.returncode:
            record.update(status="SUBMISSION_FAILED")
        else:
            record.update(parse_submission(completed.stdout), status="SUBMITTED")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        record.update(status="SUBMISSION_ID_UNKNOWN", error=str(error))
    atomic_json(path, record)
    if record["status"] != "SUBMITTED":
        raise RuntimeError(f"Slurm did not establish a job id for shard {record['shard']}; inspect {path.name} before retry")
    return record


def _state(value):
    return value.strip().split()[0].split("+")[0].upper() if value.strip() else "UNKNOWN"


def _query_scheduler(command, source, runner):
    options = {"capture_output": True, "text": True, "timeout": 15, "check": False}
    response = runner(command, **options)
    if (source == "queue" and response.returncode and not response.stdout.strip()
            and re.fullmatch(r"(?:slurm_load_jobs error:\s*)?Invalid job id specified", response.stderr.strip(), re.IGNORECASE)):
        # The controller may have purged a finished job. Query the live queue
        # without a job filter before trusting the separate accounting record.
        response = runner([arg for arg in command if not arg.startswith("--jobs=")], **options)
    return response


def _scheduler_job(receipt, offset, expected, runner):
    job = receipt["job_id"]
    cluster = ["--clusters=" + receipt["cluster"]] if receipt.get("cluster") else []
    commands = [(["squeue", "--array", "--noheader", "--jobs=" + job, "--format=%i|%T"] + cluster, "queue"),
                (["sacct", "--array", "--noheader", "--parsable2", "--jobs=" + job,
                  "--format=JobID%64,State%32,ExitCode,Cluster%128,JobIDRaw%64"] + cluster, "accounting")]
    states = {}
    errors = []
    for command, source in commands:
        try:
            response = _query_scheduler(command, source, runner)
        except (OSError, subprocess.SubprocessError) as error:
            errors.append({"job_id": job, "source": source, "error": str(error)})
            continue
        if response.returncode:
            errors.append({"job_id": job, "source": source, "error": response.stderr.strip()})
            continue
        for line in response.stdout.splitlines():
            fields = [field.strip() for field in line.split("|")]
            match = re.fullmatch(re.escape(job) + r"_(\d+)", fields[0])
            if not match:
                if source == "queue" and (fields[0] == job or fields[0].startswith(job + "_")):
                    errors.append({"job_id": job, "source": source, "error": "Unresolved array task in queue output"})
                continue
            index = int(match[1]) + offset
            if index not in expected:
                continue
            state = _state(fields[1]) if len(fields) >= 2 else "UNKNOWN"
            exit_code = fields[2] if len(fields) >= 3 else ""
            if source == "accounting" and state == "COMPLETED" and re.fullmatch(r"\d+:\d+", exit_code) and exit_code != "0:0":
                state = "FAILED"
            terminal = source == "accounting" and state in TERMINAL_STATES and bool(re.fullmatch(r"\d+:\d+", exit_code))
            row = {"state": state, "source": source, "job_id": job, "terminal": terminal}
            if terminal and len(fields) >= 5:
                account_cluster, element_job = fields[3:5]
                if receipt.get("cluster") and account_cluster and receipt["cluster"] != account_cluster:
                    row["terminal"] = False
                elif re.fullmatch(r"[A-Za-z0-9_.-]+", account_cluster) and re.fullmatch(r"[0-9]+", element_job):
                    row["owner"] = {"array_job_id": job, "array_task_id": int(match[1]),
                                    "cluster": account_cluster, "job_id": element_job}
            prior = states.setdefault(index, row)
            if prior["source"] == source and prior != row:
                states[index] = {**row, "state": "UNKNOWN", "terminal": False}
    if errors:
        for row in states.values():
            row["terminal"] = False
    return states, errors


def _scheduler_states(history, metadata, runner, selected=None):
    latest = {}
    for receipt in history:
        for index in receipt["task_indices"]:
            if selected is None or index in selected:
                latest[index] = receipt["attempt"]
    by_attempt = {}
    for index, attempt in latest.items():
        by_attempt.setdefault(attempt, set()).add(index)
    states = {}
    errors = []
    for receipt in history:
        expected = by_attempt.get(receipt["attempt"])
        if not expected or receipt["status"] != "SUBMITTED":
            continue
        offset = metadata["shards"][receipt["shard"] - 1]["start"]
        batch, batch_errors = _scheduler_job(receipt, offset, expected, runner)
        states.update(batch)
        errors.extend(batch_errors)
    return states, errors


def status(root, *, runner=None):
    root, manifest, metadata = _rendered(root)
    collection = collect_run(root, write=False)
    history = _history(root, manifest, metadata)
    states, errors = _scheduler_states(history, metadata, runner or subprocess.run)
    rows = []
    for row in collection["tasks"]:
        scheduler = states.get(row["index"], {})
        state = scheduler.get("state", "UNKNOWN")
        effective = row["status"]
        if effective != "COMPLETE":
            if state == "COMPLETED":
                effective = "MISSING_RESULT" if row["status"] == "MISSING" else row["status"]
            elif state in (TERMINAL_STATES - {"COMPLETED"}) | {"PREEMPTED"}:
                effective = "SCHEDULER_FAILED"
            elif not history:
                effective = "RENDERED"
            elif effective == "MISSING":
                effective = state
        lock = root / "tasks" / row["id"] / "dispatch.lock"
        rows.append({"id": row["id"], "index": row["index"], "scheduler_state": state,
                     "scheduler_job_id": scheduler.get("job_id"), "scheduler_source": scheduler.get("source"),
                     "scheduler_terminal_confirmed": scheduler.get("terminal", False),
                     "dispatch_lock_present": lock.exists() or lock.is_symlink(),
                     "result_status": row["status"], "status": effective})
    return {"schema": "emulateqpu.slurm_status.v1", "checked_utc": _now(),
            "status": "COMPLETE" if collection["complete"] else ("INCOMPLETE" if history else "RENDERED"),
            "complete": collection["complete"], "submission": history[-1] if history else None, "submissions": history,
            "shard_count": len(metadata["shards"]), "tasks": rows, "scheduler_errors": errors,
            "result_collection": collection, "cluster_acceptance_qualified": False}


def collect(root):
    return collect_run(root)
