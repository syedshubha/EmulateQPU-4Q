"""Immutable independent tasks, bounded subprocesses and verified collection.

Every numerical task is a separate single-process solver. Local processes and
Slurm arrays distribute independent tasks, never one state vector across nodes.
Imports and collection perform no numerical propagation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Iterable

from ._resources import MemoryObservationError, WallTimeLimitError, sample_worker_rss_bytes
from .contracts import Backend, ExecutionPlan, Study, aggregate_limit_bytes, resource_limits
from .io import atomic_json, canonical, digest, file_hash, inspect_result
from .io import read_json as _read
from .planner import _plan_validated_request

STUDY_SCHEMA = "emulateqpu.study.v1"
RUN_SCHEMA = "emulateqpu.execution.v1"
DEFAULT_MAX_RETRIES = 1
CHECKPOINT_GRACE_S = 2
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def _now():
    return datetime.now(timezone.utc).isoformat()


def task_seed(request, task_id):
    """Stable across execution order and worker count."""
    if "seed" in request:
        if type(request["seed"]) is not int or not 0 <= request["seed"] < 2**32:
            raise ValueError("Explicit seed must be an integer in 0..2**32-1")
        return request["seed"]
    return int(digest({"task_id": task_id, "request": request})[:16], 16) % 2**32


def normalize_study(request_or_study: Study | dict[str, Any] | str | Path) -> Study:
    if isinstance(request_or_study, (str, Path)):
        request_or_study = _read(request_or_study)
    value = deepcopy(request_or_study)
    if not isinstance(value, dict):
        raise ValueError("A request or study object is required")
    if value.get("schema") != STUDY_SCHEMA:
        value = {"schema": STUDY_SCHEMA, "tasks": [{"id": "task-0000", "request": value}]}
    if set(value) - {"schema", "tasks", "purpose"}:
        raise ValueError("Unknown study fields")
    if "purpose" in value and not isinstance(value["purpose"], str):
        raise ValueError("Study purpose must be text")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 10000:
        raise ValueError("A study requires 1..10000 independent tasks")
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) - {"id", "request"}:
            raise ValueError("Each task requires only id and request")
        identity = task.get("id")
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", identity):
            raise ValueError("Task ids must be safe unique labels of at most 80 characters")
        key = identity
        if os.name == "nt":
            if identity.endswith(".") or re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", identity.split(".", 1)[0], re.IGNORECASE):
                raise ValueError("Task id is not a valid Windows directory name: " + identity)
            key = identity.casefold()
        if key in seen:
            raise ValueError("Duplicate task id: " + identity)
        seen.add(key)
        if not isinstance(task.get("request"), dict):
            raise ValueError("Task request must be an object")
        # Assign a default before freezing/hashing. Explicit user seeds survive.
        task["request"]["seed"] = task_seed(task["request"], identity)
    return value


def plan_study(request_or_study: Study | dict[str, Any] | str | Path, *, backend: Backend | None = None,
               workers: int = 1, profile: dict[str, Any] | None = None,
               memory_limit_bytes: int | None = None) -> ExecutionPlan:
    """Validate every task before writing a run or launching a process."""
    study = normalize_study(request_or_study)
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("workers must be an integer in 1..64")
    if backend is None:
        backend = "serial" if workers == 1 else "local-process"
    if backend not in ("serial", "local-process", "slurm"):
        raise ValueError("backend must be serial, local-process or slurm")
    if backend == "serial" and workers != 1:
        raise ValueError("The serial backend requires workers=1")
    profile_limit_bytes = aggregate_limit_bytes(profile)
    if memory_limit_bytes is not None and (type(memory_limit_bytes) is not int or memory_limit_bytes <= 0):
        raise ValueError("memory_limit_bytes must be a positive integer")
    from .contracts import validate_request
    for task in study["tasks"]:
        task["request"] = validate_request(task["request"])
    plans = [_plan_validated_request(task["request"]) for task in study["tasks"]]
    per_worker = [plan["estimated_peak_memory_bytes"] for plan in plans]
    concurrent = min(workers, len(plans))
    coordinator_memory_bytes = 64 * 1024**2
    aggregate = coordinator_memory_bytes + sum(sorted(per_worker, reverse=True)[:concurrent])
    if profile_limit_bytes is not None and aggregate > profile_limit_bytes:
        raise ValueError("Aggregate worker memory exceeds the explicit profile budget")
    if memory_limit_bytes is not None and aggregate > memory_limit_bytes:
        raise ValueError("Aggregate worker memory exceeds the explicit study budget")
    if profile_limit_bytes is not None:
        memory_limit_bytes = min(memory_limit_bytes, profile_limit_bytes) if memory_limit_bytes is not None else profile_limit_bytes
    return {"schema": "emulateqpu.execution_plan.v1", "study": study,
            "study_sha256": digest(study), "backend": backend, "workers": workers,
            "numerical_threads_per_worker": 1, "task_count": len(plans),
            "task_plans": plans, "per_task_estimated_peak_bytes": per_worker,
            "coordinator_memory_bytes": coordinator_memory_bytes,
            "aggregate_estimated_peak_bytes": aggregate,
            "aggregate_memory_limit_bytes": memory_limit_bytes,
            "estimates_are_bounds": False, "accepted": True}


def prepare_run(request_or_study, output, *, backend=None, workers=1,
                profile=None, memory_limit_bytes=None, max_retries=DEFAULT_MAX_RETRIES):
    if type(max_retries) is not int or not 0 <= max_retries <= 3:
        raise ValueError("max_retries must be in 0..3")
    plan = plan_study(request_or_study, backend=backend, workers=workers,
                      profile=profile, memory_limit_bytes=memory_limit_bytes)
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "tasks").mkdir()
    atomic_json(root / "study.json", plan["study"])
    atomic_json(root / "plan.json", plan)
    tasks = []
    for index, task in enumerate(plan["study"]["tasks"]):
        (root / "tasks" / task["id"]).mkdir()
        tasks.append({"index": index, "id": task["id"],
                      "request_sha256": digest(task["request"]),
                      "seed": task_seed(task["request"], task["id"])})
    manifest = {"schema": RUN_SCHEMA, "created_utc": _now(), "study_sha256": plan["study_sha256"],
                "plan_sha256": digest(plan),
                "backend": plan["backend"], "workers": workers, "max_retries": max_retries,
                "tasks": tasks, "numerical_threads_per_worker": 1,
                "execution_scope": "independent tasks; each SciPy solver remains single process"}
    atomic_json(root / "run.json", manifest)
    return root


def _load_run(root):
    root = Path(root).expanduser().resolve()
    if any((root / name).is_symlink() for name in ("run.json", "study.json", "plan.json", "tasks")):
        raise ValueError("Run manifests and task directories cannot be symlinks")
    manifest = _read(root / "run.json")
    stored_study = _read(root / "study.json")
    study = normalize_study(stored_study)
    plan = _read(root / "plan.json")
    if not isinstance(manifest, dict) or not isinstance(plan, dict):
        raise ValueError("Run and plan manifests must be objects")
    if (
        manifest.get("schema") != RUN_SCHEMA
        or stored_study != study
        or manifest.get("study_sha256") != digest(study)
    ):
        raise ValueError("Run/study identity is incompatible or corrupt")
    if (
        plan.get("schema") != "emulateqpu.execution_plan.v1"
        or plan.get("study_sha256") != manifest["study_sha256"]
        or plan.get("study") != study
    ):
        raise ValueError("Plan identity is incompatible or corrupt")
    if manifest.get("backend") != plan.get("backend") or manifest.get("workers") != plan.get("workers"):
        raise ValueError("Execution allocation differs from the frozen plan")
    if type(manifest.get("max_retries")) is not int or not 0 <= manifest["max_retries"] <= 3:
        raise ValueError("Unsupported retry allowance")
    expected = [{"index": i, "id": t["id"], "request_sha256": digest(t["request"]),
                 "seed": task_seed(t["request"], t["id"])} for i, t in enumerate(study["tasks"])]
    if manifest.get("tasks") != expected:
        raise ValueError("Task mapping is incompatible or corrupt")
    plans = plan.get("task_plans")
    if (
        not isinstance(plans, list)
        or len(plans) != len(expected)
        or plan.get("task_count") != len(expected)
        or any(
            not isinstance(task_plan, dict) or task_plan.get("request_sha256") != identity["request_sha256"]
            for task_plan, identity in zip(plans, expected)
        )
    ):
        raise ValueError("Task plan identities are incompatible or corrupt")
    if "plan_sha256" in manifest and manifest["plan_sha256"] != digest(plan):
        raise ValueError("Execution plan digest differs from the saved run manifest")
    if any((root / "tasks" / task["id"]).is_symlink() for task in study["tasks"]):
        raise ValueError("Task directories cannot be symlinks")
    return root, manifest, study, plan


def _file_manifest(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Result directories cannot be symlinks")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Result symlinks are not supported")
        if path.is_file():
            rows.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
                         "sha256": file_hash(path)})
    return rows


def _result_complete(output, expected_request_sha256=None):
    result_path = Path(output) / "result.json"
    if not result_path.is_file():
        raise ValueError("Task has no result.json manifest")
    result = inspect_result(result_path)
    if result.get("status") != "PASS" or result.get("complete") is not True:
        raise ValueError("Task result does not declare successful computational completion")
    if expected_request_sha256 is not None and result.get("request_sha256") != expected_request_sha256:
        raise ValueError("Result belongs to a different immutable request")
    return result


def _worker(request_file, output, receipt_file, task_id, request_sha256, seed, *,
            deadline=None, resume_from=None):
    """Entry used by controlled subprocesses; imports physics only here."""
    receipt = {"schema": "emulateqpu.task_attempt.v1", "task_id": task_id,
               "request_sha256": request_sha256, "seed": seed, "started_utc": _now(),
               "status": "RUNNING", "numerical_threads": 1}
    started = time.monotonic()
    try:
        if deadline is not None and started >= deadline:
            raise WallTimeLimitError("Wall budget reached before worker startup")
        request = _read(request_file)
        if digest(request) != request_sha256:
            raise ValueError("Task request bytes changed")
        from .api import _run_request
        # Task seed was assigned before its immutable request hash was computed.
        _run_request(request, output, deadline=deadline, resume_from=resume_from)
        result = _result_complete(output, request_sha256)
        receipt.update(status="COMPLETE", result_schema=result.get("schema"),
                       files=_file_manifest(output))
        code = 0
    except BaseException as error:
        receipt.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        if isinstance(error, WallTimeLimitError):
            receipt["failure_reason"] = "WALL_TIME_LIMIT"
        elif isinstance(error, MemoryObservationError):
            receipt["failure_reason"] = "MEMORY_OBSERVATION_FAILED"
        code = 1
    receipt.update(ended_utc=_now(), elapsed_s=time.monotonic() - started)
    atomic_json(receipt_file, receipt)
    return code


def _terminate(process, *, grace_s=2):
    if process.poll() is not None:
        return
    if os.name == "posix":
        _signal_group(process, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            _signal_group(process, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def _signal_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        # The process can exit between poll() and signaling on Darwin.
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            raise error from None


def _latest_checkpoint(previous_attempts):
    for attempt in reversed(previous_attempts):
        output = attempt / "output"
        pointer = output / "checkpoint.json"
        if output.is_symlink() or pointer.is_symlink():
            raise ValueError("Checkpoint directories and pointers cannot be symlinks")
        if pointer.exists():
            return output
    return None


def execute_task(root, index, *, retry=False):
    """Execute one stable array index. Duplicate concurrent dispatch is rejected."""
    root, manifest, study, _ = _load_run(root)
    return _execute_task(root, manifest, study, index, retry=retry)


def _execute_task(root, manifest, study, index, *, retry):
    if type(index) is not int or not 0 <= index < len(study["tasks"]):
        raise ValueError("Task index outside the immutable array map")
    task = study["tasks"][index]
    identity = manifest["tasks"][index]
    task_root = root / "tasks" / task["id"]
    if task_root.is_symlink():
        raise ValueError("Task directories cannot be symlinks")
    memory_limit, wall_limit = resource_limits(task["request"]["resources"])
    lock = task_root / "dispatch.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("Task is already dispatched; inspect its preserved attempt") from error
    process = None
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write((canonical({"controller_pid": os.getpid(), "created_utc": _now(),
                                     "task_id": task["id"]}) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        previous = sorted(task_root.glob("attempt_*"))
        if any(attempt.is_symlink() for attempt in previous):
            raise ValueError("Attempt directories cannot be symlinks")
        for attempt in previous:
            receipt = attempt / "attempt.json"
            if receipt.is_file() and _read(receipt).get("status") == "COMPLETE":
                raise ValueError("Task already completed; duplicate dispatch rejected")
        if previous and not retry:
            raise ValueError("A prior attempt exists; use explicit bounded retry/resume")
        if len(previous) > manifest["max_retries"]:
            raise ValueError("Task retry allowance exhausted")
        checkpoint = _latest_checkpoint(previous) if task["request"].get("engine") == "chain" else None
        attempt = task_root / ("attempt_%04d" % (len(previous) + 1))
        attempt.mkdir()
        request_file = attempt / "request.json"
        atomic_json(request_file, task["request"])
        receipt_path = attempt / "attempt.json"
        atomic_json(receipt_path, {"schema": "emulateqpu.task_attempt.v1", "status": "RUNNING", **identity,
                                   "started_utc": _now()})
        environment = dict(os.environ)
        environment.update({name: "1" for name in THREAD_VARIABLES})
        environment["PYTHONHASHSEED"] = str(identity["seed"])
        environment["EMULATEQPU_TASK_SEED"] = str(identity["seed"])
        cmd = [sys.executable, "-m", "emulateqpu.execution", "--worker", str(request_file),
               str(attempt / "output"), str(receipt_path), task["id"], identity["request_sha256"], str(identity["seed"])]
        started = time.monotonic()
        deadline = started + wall_limit
        kill_deadline = deadline + CHECKPOINT_GRACE_S
        cmd.extend(("--deadline", str(deadline)))
        if checkpoint is not None:
            cmd.extend(("--resume-from", str(checkpoint)))
        peak = None
        reason = None
        observation_error = None
        with (attempt / "stdout.log").open("wb") as stdout, (attempt / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=environment,
                                       start_new_session=(os.name == "posix"))
            atomic_json(attempt / "dispatch.json", {"controller_pid": os.getpid(), "worker_pid": process.pid,
                        "command": cmd, "started_utc": _now(), "request_sha256": identity["request_sha256"]})
            try:
                while process.poll() is None:
                    if time.monotonic() >= kill_deadline:
                        reason = "WALL_TIME_LIMIT"
                        _terminate(process, grace_s=0)
                        break
                    observed = sample_worker_rss_bytes(process)
                    if observed is None:
                        break
                    peak = observed if peak is None else max(peak, observed)
                    if peak > memory_limit:
                        reason = "MEMORY_LIMIT"
                        _terminate(process)
                        break
                    time.sleep(max(0, min(0.05, kill_deadline - time.monotonic())))
            except MemoryObservationError as error:
                reason = "MEMORY_OBSERVATION_FAILED"
                observation_error = str(error)
                _terminate(process)
            except BaseException:
                _terminate(process)
                reason = "INTERRUPTED"
                raise
            finally:
                record = _read(receipt_path)
                record.update(returncode=process.poll(), observed_peak_rss_bytes=peak,
                              controller_elapsed_s=time.monotonic() - started,
                              resource_limits={"memory_bytes": memory_limit, "wall_time_s": wall_limit},
                              ended_utc=_now())
                if observation_error is not None:
                    record["error"] = observation_error
                if checkpoint is not None:
                    record["resumed_from_attempt"] = checkpoint.parent.name
                if reason or process.poll() != 0 or record.get("status") != "COMPLETE":
                    record.update(status="INTERRUPTED" if reason == "INTERRUPTED" else "FAILED",
                                  failure_reason=reason or record.get("failure_reason", "WORKER_FAILED"))
                atomic_json(receipt_path, record)
        return record
    finally:
        if process is not None and process.poll() is None:
            _terminate(process)
        lock.unlink(missing_ok=True)


def collect_run(root, *, write=True):
    """Verify all expected outputs, preserving partial/failed/duplicate status."""
    root, manifest, study, plan = _load_run(root)
    expected = {t["id"] for t in study["tasks"]}
    unexpected = sorted(p.name for p in (root / "tasks").iterdir() if p.name not in expected)
    rows = []
    for task, identity in zip(study["tasks"], manifest["tasks"]):
        row = {"id": task["id"], "index": identity["index"], "status": "MISSING", "attempts": []}
        successes = []
        for attempt in sorted((root / "tasks" / task["id"]).glob("attempt_*")):
            try:
                if attempt.is_symlink():
                    raise ValueError("Attempt directories cannot be symlinks")
                receipt = _read(attempt / "attempt.json")
                if receipt.get("task_id", receipt.get("id")) != task["id"] or receipt.get("request_sha256") != identity["request_sha256"]:
                    raise ValueError("Attempt belongs to a different task or input")
                if digest(_read(attempt / "request.json")) != identity["request_sha256"]:
                    raise ValueError("Saved request changed")
                item = {"attempt": attempt.name, "status": receipt.get("status", "INCOMPLETE")}
                for key in ("failure_reason", "error"):
                    if key in receipt:
                        item[key] = receipt[key]
                if item["status"] == "COMPLETE":
                    _result_complete(attempt / "output", identity["request_sha256"])
                    if _file_manifest(attempt / "output") != receipt.get("files"):
                        raise ValueError("Result inventory or checksum differs")
                    successes.append(attempt.name)
                row["attempts"].append(item)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row["attempts"].append({"attempt": attempt.name, "status": "CORRUPT", "error": str(error)})
        statuses = [a["status"] for a in row["attempts"]]
        if "CORRUPT" in statuses:
            row["status"] = "CORRUPT"
        elif len(successes) > 1:
            row["status"] = "DUPLICATE"
        elif successes:
            row.update(status="COMPLETE", successful_attempt=successes[0])
        elif statuses:
            row["status"] = statuses[-1]
        rows.append(row)
    complete = not unexpected and all(t["status"] == "COMPLETE" for t in rows)
    summary = {"schema": "emulateqpu.collection.v1", "study_sha256": manifest["study_sha256"],
               "collected_utc": _now(), "status": "COMPLETE" if complete else "INCOMPLETE",
               "complete": complete, "expected_tasks": len(rows),
               "complete_tasks": sum(t["status"] == "COMPLETE" for t in rows),
               "unexpected_task_directories": unexpected, "tasks": rows,
               "scientific_qualification": "Separate from computational completion; inspect each result"}
    if write:
        atomic_json(root / "collection.json", summary)
    return summary


def resume_run(root, *, backend=None, workers=None):
    root, manifest, study, plan = _load_run(root)
    backend = manifest["backend"] if backend is None else backend
    workers = manifest["workers"] if workers is None else workers
    if backend not in ("serial", "local-process"):
        raise ValueError("Use explicit Slurm resubmission for cluster retries")
    # Recheck current resource policy before any new allocation.
    plan_study(study, backend=backend, workers=workers,
               memory_limit_bytes=plan.get("aggregate_memory_limit_bytes"))
    collection = collect_run(root)
    if collection["unexpected_task_directories"] or any(t["status"] in ("CORRUPT", "DUPLICATE") for t in collection["tasks"]):
        raise ValueError("Cannot resume corrupt or duplicate outputs")
    pending = [t["index"] for t in collection["tasks"] if t["status"] != "COMPLETE"]
    if any((root / "tasks" / study["tasks"][i]["id"] / "dispatch.lock").exists() for i in pending):
        raise ValueError("An unfinished dispatch lock requires inspection; no concurrent resume")
    return _dispatch_tasks(root, manifest, study, workers, pending)


def _dispatch_tasks(root, manifest, study: Study, workers: int, pending: Iterable[int]):
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_execute_task, root, manifest, study, index, retry=True): index for index in pending
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append({"index": futures[future], "error": str(error)})
    summary = collect_run(root)
    if errors:
        summary["dispatch_errors"] = errors
        atomic_json(root / "collection.json", summary)
    return summary


def run_study(request_or_study, output, *, backend=None, workers=1,
              profile=None, memory_limit_bytes=None, max_retries=DEFAULT_MAX_RETRIES):
    if backend not in (None, "serial", "local-process"):
        raise ValueError("Render and explicitly submit Slurm plans with emulateqpu hpc")
    root = prepare_run(request_or_study, output, backend=backend, workers=workers,
                       profile=profile, memory_limit_bytes=memory_limit_bytes, max_retries=max_retries)
    root, manifest, study, _ = _load_run(root)
    return _dispatch_tasks(root, manifest, study, workers, range(len(study["tasks"])))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Internal immutable task worker")
    parser.add_argument("--worker", nargs=6, metavar=("REQUEST", "OUTPUT", "RECEIPT", "TASK", "HASH", "SEED"))
    parser.add_argument("--run-task", nargs=2, metavar=("RUN_DIRECTORY", "INDEX"))
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        request, output, receipt, task, identity, seed = args.worker
        return _worker(request, output, receipt, task, identity, int(seed),
                       deadline=args.deadline, resume_from=args.resume_from)
    if args.run_task:
        root, index = args.run_task
        record = execute_task(root, int(index), retry=True)
        return 0 if record["status"] == "COMPLETE" else 2
    parser.error("An internal worker mode is required")


if __name__ == "__main__":
    raise SystemExit(main())
