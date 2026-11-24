"""Bounded independent-task and mock-scheduler acceptance checks."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from emulateqpu import execution as ex
from emulateqpu import hpc
from emulateqpu import palace
from emulateqpu import _resources


def _fixture_plan(request):
    return {"estimated_peak_memory_bytes": 10000, "request_sha256": ex.digest(request)}


class WorkerMemoryTests(unittest.TestCase):
    def test_zero_rss_is_resampled_once_for_live_worker(self):
        child = Mock(spec=subprocess.Popen, pid=123)
        child.poll.return_value = None
        child.wait.side_effect = subprocess.TimeoutExpired("worker", 0.05)
        with patch.object(_resources, "process_tree_rss_bytes", side_effect=(0, 4096)) as sample:
            self.assertEqual(_resources.sample_worker_rss_bytes(child), 4096)
        self.assertEqual(sample.call_count, 2)
        child.wait.assert_called_once_with(timeout=0.05)

    def test_zero_rss_returns_none_for_confirmed_exit(self):
        child = Mock(spec=subprocess.Popen, pid=123)
        child.wait.return_value = 0
        with patch.object(_resources, "process_tree_rss_bytes", return_value=0) as sample:
            self.assertIsNone(_resources.sample_worker_rss_bytes(child))
        sample.assert_called_once_with(123)

    def test_persistent_zero_rss_fails_for_live_worker(self):
        child = Mock(spec=subprocess.Popen, pid=123)
        child.poll.return_value = None
        child.wait.side_effect = subprocess.TimeoutExpired("worker", 0.05)
        with patch.object(_resources, "process_tree_rss_bytes", return_value=0) as sample:
            with self.assertRaisesRegex(_resources.MemoryObservationError, "Worker RSS is unavailable"):
                _resources.sample_worker_rss_bytes(child)
        self.assertEqual(sample.call_count, 2)

    def test_process_disappearance_waits_for_delayed_reap(self):
        child = Mock(spec=subprocess.Popen, pid=123)
        child.wait.side_effect = [subprocess.TimeoutExpired("worker", 0.05), 0]
        with patch.object(_resources, "process_tree_rss_bytes",
                          side_effect=_resources.psutil.NoSuchProcess(123)) as sample:
            self.assertIsNone(_resources.sample_worker_rss_bytes(child))
        self.assertEqual(sample.call_count, 2)
        self.assertEqual([call.kwargs["timeout"] for call in child.wait.call_args_list], [0.05, 0.2])

    def test_process_disappearance_fails_if_exit_cannot_be_confirmed(self):
        child = Mock(spec=subprocess.Popen, pid=123)
        child.wait.side_effect = subprocess.TimeoutExpired("worker", 0.2)
        with patch.object(_resources, "process_tree_rss_bytes",
                          side_effect=_resources.psutil.NoSuchProcess(123)):
            with self.assertRaisesRegex(_resources.MemoryObservationError, "Cannot sample worker RSS"):
                _resources.sample_worker_rss_bytes(child)

    def test_signal_permission_race_requires_confirmed_exit(self):
        child = Mock(spec=subprocess.Popen, pid=123)
        child.wait.return_value = 0
        with patch.object(ex.os, "killpg", side_effect=PermissionError("cannot signal"), create=True):
            ex._signal_group(child, ex.signal.SIGTERM)
            child.wait.assert_called_once_with(timeout=0.2)
            child.wait.side_effect = subprocess.TimeoutExpired("worker", 0.2)
            with self.assertRaisesRegex(PermissionError, "cannot signal"):
                ex._signal_group(child, ex.signal.SIGTERM)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="emulateqpu-execution-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = {"schema": "fixture", "resources": {"memory_mb": 128, "wall_time_s": 10}}
        self.study = {"schema": ex.STUDY_SCHEMA, "tasks": [
            {"id": "left", "request": deepcopy(self.request)},
            {"id": "right", "request": deepcopy(self.request)}]}

    def _prepare(self, path="run", *, study=None, retries=1, backend="serial", memory_limit_bytes=None):
        with patch("emulateqpu.contracts.validate_request", side_effect=deepcopy), \
             patch.object(ex, "_plan_validated_request", side_effect=_fixture_plan):
            return ex.prepare_run(study or self.study, self.root / path, max_retries=retries, backend=backend, memory_limit_bytes=memory_limit_bytes)

    def _success(self, root, task_index, number=1):
        _, manifest, study, _ = ex._load_run(root)
        task = study["tasks"][task_index]
        identity = manifest["tasks"][task_index]
        attempt = root / "tasks" / task["id"] / ("attempt_%04d" % number)
        attempt.mkdir()
        ex.atomic_json(attempt / "request.json", task["request"])
        output = attempt / "output"
        output.mkdir()
        ex.atomic_json(output / "result.json", {"schema": "emulateqpu.result.v1", "status": "PASS", "complete": True,
                                              "request_sha256": identity["request_sha256"], "payload_sha256": {}})
        (output / "states.bin").write_bytes(b"preserved complex states")
        ex.atomic_json(attempt / "attempt.json", {"task_id": task["id"],
            "request_sha256": identity["request_sha256"], "seed": identity["seed"],
            "status": "COMPLETE", "files": ex._file_manifest(output)})
        return attempt

    def test_seed_and_task_mapping_survive_reordering(self):
        normal = ex.normalize_study(self.study)
        reordered = ex.normalize_study({"schema": ex.STUDY_SCHEMA, "tasks": list(reversed(self.study["tasks"]))})
        seeds = {t["id"]: t["request"]["seed"] for t in normal["tasks"]}
        self.assertEqual(seeds, {t["id"]: t["request"]["seed"] for t in reordered["tasks"]})
        self.assertNotEqual(seeds["left"], seeds["right"])
        self.assertNotIn("seed", self.study["tasks"][0]["request"])
        explicit = deepcopy(self.request); explicit["seed"] = 7
        self.assertEqual(ex.task_seed(explicit, "unrelated"), 7)

    def test_duplicates_and_path_task_ids_rejected(self):
        for identity in ("../escape", "has space", ""):
            invalid = deepcopy(self.study); invalid["tasks"][0]["id"] = identity
            with self.assertRaises(ValueError): ex.normalize_study(invalid)
        invalid = deepcopy(self.study); invalid["tasks"][1]["id"] = "left"
        with self.assertRaises(ValueError): ex.normalize_study(invalid)

    def test_aggregate_rejection_precedes_creation(self):
        with patch("emulateqpu.contracts.validate_request", side_effect=deepcopy), \
             patch.object(ex, "_plan_validated_request", side_effect=_fixture_plan):
            with self.assertRaisesRegex(ValueError, "Aggregate"):
                ex.prepare_run(self.study, self.root / "rejected", backend="local-process", workers=2,
                               memory_limit_bytes=15000)
        self.assertFalse((self.root / "rejected").exists())

    def test_collection_missing_failed_complete_and_corrupt(self):
        root = self._prepare()
        self.assertEqual(ex.collect_run(root)["tasks"][0]["status"], "MISSING")
        left = self._success(root, 0)
        self.assertFalse(ex.collect_run(root)["complete"])
        self._success(root, 1)
        self.assertTrue(ex.collect_run(root)["complete"])
        (left / "output" / "states.bin").write_bytes(b"corrupted")
        summary = ex.collect_run(root)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["tasks"][0]["status"], "CORRUPT")

    def test_failed_result_is_not_a_successful_complete_attempt(self):
        root = self._prepare()
        attempt = self._success(root, 0)
        for status in ("FAIL", "COMPLETED"):
            with self.subTest(status=status):
                ex.atomic_json(attempt / "output/result.json", {
                    "schema": "emulateqpu.result.v1", "status": status, "complete": True,
                })
                with self.assertRaisesRegex(ValueError, "successful"):
                    ex._result_complete(attempt / "output")

    def test_duplicate_success_is_never_collected_as_complete(self):
        root = self._prepare()
        self._success(root, 0)
        self._success(root, 0, 2)
        self._success(root, 1)
        self.assertEqual(ex.collect_run(root)["tasks"][0]["status"], "DUPLICATE")
        with self.assertRaises(ValueError):
            ex.execute_task(root, 0)

    def test_changed_identity_and_unexpected_outputs_rejected(self):
        root = self._prepare()
        (root / "tasks" / "unknown").mkdir()
        self.assertEqual(ex.collect_run(root)["unexpected_task_directories"], ["unknown"])
        saved = ex._read(root / "study.json"); saved["tasks"][0]["request"]["seed"] += 1
        ex.atomic_json(root / "study.json", saved)
        with self.assertRaisesRegex(ValueError, "identity"):
            ex.collect_run(root)

    def test_mismatched_task_plan_is_rejected_before_dispatch(self):
        root = self._prepare()
        saved = ex._read(root / "plan.json")
        saved["task_plans"].reverse()
        ex.atomic_json(root / "plan.json", saved)
        with self.assertRaisesRegex(ValueError, "Task plan identities"):
            ex.execute_task(root, 0)
        self.assertEqual(list((root / "tasks/left").iterdir()), [])

    def test_plan_digest_guards_aggregate_budget_on_new_runs(self):
        root = self._prepare()
        manifest = ex._read(root / "run.json")
        saved = ex._read(root / "plan.json")
        self.assertEqual(manifest["plan_sha256"], ex.digest(saved))
        saved["aggregate_memory_limit_bytes"] = 1024**3
        ex.atomic_json(root / "plan.json", saved)
        with self.assertRaisesRegex(ValueError, "plan digest differs"):
            ex.collect_run(root)

    def test_failed_attempt_remains_and_retry_is_bounded(self):
        root = self._prepare(study={"schema": ex.STUDY_SCHEMA, "tasks": [self.study["tasks"][0]]}, retries=1)
        # A schema-invalid request fails inside the real fresh worker process.
        record = ex.execute_task(root, 0)
        self.assertEqual(record["status"], "FAILED")
        self.assertTrue((root / "tasks/left/attempt_0001/stderr.log").exists())
        record = ex.execute_task(root, 0, retry=True)
        self.assertEqual(record["status"], "FAILED")
        with self.assertRaisesRegex(ValueError, "allowance"):
            ex.execute_task(root, 0, retry=True)
        self.assertTrue((root / "tasks/left/attempt_0001/attempt.json").is_file())
        self.assertEqual(ex.collect_run(root)["tasks"][0]["status"], "FAILED")

    def test_explicit_zero_retries_is_preserved(self):
        root = self._prepare(study={"schema": ex.STUDY_SCHEMA, "tasks": [self.study["tasks"][0]]}, retries=0)
        self.assertEqual(ex.execute_task(root, 0)["status"], "FAILED")
        with self.assertRaisesRegex(ValueError, "allowance"):
            ex.execute_task(root, 0, retry=True)
        self.assertFalse((root / "tasks/left/attempt_0002").exists())

    def test_dispatch_lock_prevents_concurrent_task(self):
        root = self._prepare()
        (root / "tasks/left/dispatch.lock").write_text("existing worker")
        with self.assertRaisesRegex(ValueError, "already dispatched"):
            ex.execute_task(root, 0)

    def test_symlinked_task_directory_is_rejected_before_dispatch(self):
        root = self._prepare()
        task = root / "tasks/left"
        task.rmdir()
        elsewhere = self.root / "elsewhere"; elsewhere.mkdir()
        try:
            task.symlink_to(elsewhere, target_is_directory=True)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows runner lacks symlink creation privilege")
            raise
        with self.assertRaisesRegex(ValueError, "symlinks"):
            ex.execute_task(root, 0)
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_worker_memory_guard_preserves_resource_stop(self):
        request = deepcopy(self.request); request["resources"]["memory_mb"] = 1
        root = self._prepare(study={"schema": ex.STUDY_SCHEMA, "tasks": [{"id": "small-budget", "request": request}]}, memory_limit_bytes=128*1024**2)
        result = ex.execute_task(root, 0)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["failure_reason"], "MEMORY_LIMIT")

    def test_worker_wall_guard_preserves_resource_stop(self):
        request = deepcopy(self.request); request["resources"]["wall_time_s"] = .001
        root = self._prepare(study={"schema": ex.STUDY_SCHEMA, "tasks": [{"id": "short-budget", "request": request}]})
        result = ex.execute_task(root, 0)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["failure_reason"], "WALL_TIME_LIMIT")

    def test_memory_observation_failure_stops_worker_and_is_reported(self):
        root = self._prepare()
        with patch.object(ex, "sample_worker_rss_bytes", side_effect=ex.MemoryObservationError("reading denied")):
            result = ex.execute_task(root, 0)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["failure_reason"], "MEMORY_OBSERVATION_FAILED")
        self.assertEqual(result["error"], "reading denied")
        self.assertIsNone(result["observed_peak_rss_bytes"])
        self.assertFalse((root / "tasks/left/dispatch.lock").exists())


class SlurmTests(unittest.TestCase):
    setUp = ExecutionTests.setUp
    _success = ExecutionTests._success
    def _render(self, path="slurm run", *, study=None, **site):
        profile = {"schema": hpc.PROFILE_SCHEMA, "shared_filesystem": True,
                   "shared_root": str(self.root), "memory_mb": 128, "wall_time": "00:01:00",
                   "array_concurrency": 2, "cpus_per_task": 1, "max_retries": 1,
                   "partition": "short", "account": "project", "environment": {"CASE": "a b"}, **site}
        with patch("emulateqpu.contracts.validate_request", side_effect=deepcopy), \
             patch.object(ex, "_plan_validated_request", side_effect=_fixture_plan):
            metadata = hpc.render(self.study if study is None else study, profile, self.root / path)
        return self.root / path, metadata

    def test_rendering_is_literal_and_array_mapping_stable(self):
        root, metadata = self._render()
        script = (root / "job.sbatch").read_text()
        self.assertIn("#SBATCH --array=0-1%2", script)
        self.assertIn('"$SLURM_ARRAY_TASK_ID"', script)
        self.assertIn("#SBATCH --cpus-per-task=1", script)
        self.assertIn("export OMP_NUM_THREADS=1", script)
        self.assertTrue(all("$" not in line for line in script.splitlines() if line.startswith("#SBATCH")))
        self.assertEqual([t["id"] for t in metadata["task_map"]], ["left", "right"])
        self.assertEqual(hpc.status(root)["status"], "RENDERED")
        with self.assertRaises(ValueError): hpc.validate_profile({"shared_filesystem": False})

    def test_submission_cluster_suffix_and_duplicate_protection(self):
        root, _ = self._render()
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "1234;cluster-a\n", "")
        record = hpc.submit(root, runner=runner)
        self.assertEqual(record["job_id"], "1234")
        self.assertEqual(record["cluster"], "cluster-a")
        self.assertEqual(calls[0][:2], ["sbatch", "--parsable"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            hpc.submit(root, runner=runner)
        with self.assertRaises(ValueError): hpc.parse_submission("Submitted batch job 1234")

    def test_concurrent_submission_lock_prevents_scheduler_call(self):
        root, _ = self._render()
        (root / "submission.lock").write_text("existing controller")
        with patch("subprocess.run") as runner:
            with self.assertRaisesRegex(ValueError, "in progress"):
                hpc.submit(root)
            runner.assert_not_called()

    def test_scheduler_completion_needs_result_manifest(self):
        root, _ = self._render()
        def runner(command, **kwargs):
            if command[0] == "sbatch": out = "1234\n"
            elif command[0] == "squeue": out = ""
            else: out = "1234_0|COMPLETED|0:0\n1234_1|FAILED|1:0\n1234.batch|COMPLETED|0:0\n"
            return subprocess.CompletedProcess(command, 0, out, "")
        hpc.submit(root, runner=runner)
        state = hpc.status(root, runner=runner)
        self.assertFalse(state["complete"])
        self.assertEqual(state["tasks"][0]["status"], "MISSING_RESULT")
        self.assertEqual(state["tasks"][1]["status"], "SCHEDULER_FAILED")
        self._success(root, 0); self._success(root, 1)
        self.assertTrue(hpc.status(root, runner=runner)["complete"])

    def test_absent_queue_and_accounting_is_unknown(self):
        root, _ = self._render()
        hpc.submit(root, runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "1234\n", ""))
        state = hpc.status(root, runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""))
        self.assertEqual(state["tasks"][0]["scheduler_state"], "UNKNOWN")
        self.assertFalse(state["complete"])
        with self.assertRaisesRegex(ValueError, "termination"):
            hpc.submit(root, retry=True, runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""))

    def test_bounded_retry_preserves_submission_and_array_mapping(self):
        root, _ = self._render()
        def initial(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "1234\n", "")
        hpc.submit(root, runner=initial)
        self._success(root, 0)
        calls = []
        def retry(command, **kwargs):
            calls.append(command)
            if command[0] == "squeue": out = ""
            elif command[0] == "sacct": out = "1234_0|COMPLETED|0:0\n1234_1|FAILED|1:0\n"
            else: out = "1235;cluster-b\n"
            return subprocess.CompletedProcess(command, 0, out, "")
        record = hpc.submit(root, retry=True, runner=retry)
        self.assertEqual(record["job_id"], "1235")
        self.assertIn("--array=1%2", calls[-1])
        self.assertTrue((root / "submission_0001.json").is_file())
        self.assertTrue((root / "submission_0002.json").is_file())
        with self.assertRaisesRegex(ValueError, "allowance"):
            hpc.submit(root, retry=True, runner=retry)

    def test_retry_requires_terminal_accounting_and_no_active_queue_entry(self):
        root, _ = self._render()
        hpc.submit(root, runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "1234\n", ""))
        self._success(root, 0)
        for queued, accounting, queue_error in [
            ("1234_1|RESIZING\n", "1234_1|FAILED|1:0\n", False),
            ("1234_[1]|PENDING\n", "1234_1|FAILED|1:0\n", False),
            ("", "1234_1|PREEMPTED|0:0\n", False),
            ("", "1234_1|FAILED|1:0\n", True),
            ("", "1234_1|FAILED|\n", False),
        ]:
            with self.subTest(queued=queued, accounting=accounting, queue_error=queue_error):
                def runner(command, **kwargs):
                    self.assertNotEqual(command[0], "sbatch")
                    if command[0] == "squeue":
                        return subprocess.CompletedProcess(command, int(queue_error), queued, "queue error" if queue_error else "")
                    return subprocess.CompletedProcess(command, 0, "1234_0|COMPLETED|0:0\n" + accounting, "")
                with self.assertRaisesRegex(ValueError, "termination.*right"):
                    hpc.submit(root, retry=True, runner=runner)
                self.assertFalse((root / "submission_0002.json").exists())

    def test_purged_queue_job_uses_unfiltered_queue_before_accounting(self):
        root, _ = self._render()
        hpc.submit(root, runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "1234\n", ""))
        self._success(root, 0)
        queued = "1234_1|RUNNING\n"
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            if command[0] == "squeue" and "--jobs=1234" in command:
                return subprocess.CompletedProcess(command, 1, "", "slurm_load_jobs error: Invalid job id specified\n")
            output = {"squeue": queued, "sacct": "1234_1|FAILED|1:0\n", "sbatch": "1235\n"}[command[0]]
            return subprocess.CompletedProcess(command, 0, output, "")
        with self.assertRaisesRegex(ValueError, "termination"):
            hpc.submit(root, retry=True, runner=runner)
        self.assertEqual([command[0] for command in calls], ["squeue", "squeue", "sacct"])
        self.assertIn("--array", calls[1])
        self.assertNotIn("--jobs=1234", calls[1])
        queued = "5678_1|RUNNING\n"
        receipt = hpc.submit(root, retry=True, runner=runner)
        self.assertEqual(receipt["task_indices"], [1])

    def test_terminal_task_with_preserved_dispatch_lock_does_not_consume_retry(self):
        root, _ = self._render()
        hpc.submit(root, runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "1234\n", ""))
        self._success(root, 0)
        lock = root / "tasks/right/dispatch.lock"
        lock.write_text("preserved controller")
        def runner(command, **kwargs):
            self.assertNotEqual(command[0], "sbatch")
            output = "1234_1|TIMEOUT|0:15\n" if command[0] == "sacct" else ""
            return subprocess.CompletedProcess(command, 0, output, "")
        state = hpc.status(root, runner=runner)["tasks"][1]
        self.assertTrue(state["scheduler_terminal_confirmed"])
        self.assertTrue(state["dispatch_lock_present"])
        with self.assertRaisesRegex(ValueError, "Dispatch lock remains for right"):
            hpc.submit(root, retry=True, runner=runner)
        self.assertEqual(lock.read_text(), "preserved controller")
        self.assertFalse((root / "submission_0002.json").exists())

    def test_dispatch_lock_records_slurm_owner_before_worker_launch(self):
        root, metadata = self._render(max_array_size=1)
        owner = {"job_id": "19", "array_job_id": "17", "array_task_id": 0, "cluster": "example"}
        environment = {"SLURM_JOB_ID": "19", "SLURM_ARRAY_JOB_ID": "17",
                       "SLURM_ARRAY_TASK_ID": "0", "SLURM_CLUSTER_NAME": "example"}
        lock = root / "tasks/right/dispatch.lock"
        def stop_after_lock(*args, **kwargs):
            stored = ex._read(lock)
            self.assertEqual(stored["slurm"], owner)
            self.assertEqual(stored["request_sha256"], metadata["task_map"][1]["request_sha256"])
            raise OSError("fixture stops before worker launch")
        with patch.dict(os.environ, environment), patch.object(ex.subprocess, "Popen", side_effect=stop_after_lock):
            with self.assertRaisesRegex(OSError, "fixture stops"):
                ex.execute_task(root, 1)
        self.assertFalse(lock.exists())

    def test_terminal_owned_lock_is_archived_for_exact_array_element(self):
        root, metadata = self._render(max_array_size=1)
        ids = iter((17, 18, 20))
        def runner(command, **kwargs):
            if command[0] == "sacct":
                output = "18_0|CANCELLED by 1000|0:9|example|19\n"
            elif command[0] == "squeue":
                output = ""
            else:
                job = next(ids)
                if job == 20:
                    self.assertFalse(lock.exists())
                    self.assertEqual(archive.read_bytes(), payload)
                output = f"{job};example\n"
            return subprocess.CompletedProcess(command, 0, output, "")
        hpc.submit(root, runner=runner)
        self._success(root, 0)
        lock = root / "tasks/right/dispatch.lock"
        archive = root / "tasks/right/dispatch-lock-example-19.json"
        owner = {"job_id": "19", "array_job_id": "18", "array_task_id": 0, "cluster": "example"}
        lock_record = {"task_id": "right", "request_sha256": metadata["task_map"][1]["request_sha256"], "slurm": owner}
        for key, wrong in (("job_id", "18"), ("array_job_id", "17"), ("array_task_id", 1), ("cluster", "other")):
            with self.subTest(wrong_owner=key):
                ex.atomic_json(lock, {**lock_record, "slurm": {**owner, key: wrong}})
                with self.assertRaisesRegex(ValueError, "Dispatch lock remains for right"):
                    hpc.submit(root, retry=True, runner=runner)
                self.assertFalse(archive.exists())
                self.assertFalse((root / "submission_0003.json").exists())
        ex.atomic_json(lock, lock_record)
        payload = lock.read_bytes()
        receipt = hpc.submit(root, retry=True, runner=runner)
        self.assertEqual(receipt["task_indices"], [1])
        self.assertIn("--array=0%2", receipt["command"])

    def test_sharded_arrays_keep_global_identities_and_retry_only_missing_indices(self):
        study = {**self.study, "tasks": [{"id": f"task-{i}", "request": self.request} for i in range(5)]}
        root, metadata = self._render(study=study, max_array_size=2, dependency="afterok:77")
        self.assertEqual([(shard["start"], shard["stop"]) for shard in metadata["shards"]], [(0, 2), (2, 4), (4, 5)])
        self.assertEqual([entry["id"] for entry in metadata["task_map"]], [f"task-{i}" for i in range(5)])
        self.assertIn('"$((SLURM_ARRAY_TASK_ID + 2))"', (root / "job-0002.sbatch").read_text())
        ids = iter((100, 101, 102, 103, 104))
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            if command[0] == "sbatch":
                output = f"{next(ids)};cluster-a\n"
            elif command[0] == "squeue":
                output = ""
            else:
                job = next(argument.split("=")[1] for argument in command if argument.startswith("--jobs="))
                output = f"{job}_0|COMPLETED|0:0\n{job}_1|TIMEOUT|0:15\n"
            return subprocess.CompletedProcess(command, 0, output, "")
        initial = hpc.submit(root, runner=runner)["submissions"]
        self.assertEqual([receipt["task_indices"] for receipt in initial], [[0, 1], [2, 3], [4]])
        self.assertIn("--array=0-0%2", calls[2])
        self.assertIn("--dependency=afterany:100,afterok:77", calls[1])
        self.assertIn("--dependency=afterany:101,afterok:77", calls[2])
        self.assertIn("--clusters=cluster-a", calls[2])
        for i in (0, 2, 4):
            self._success(root, i)
        retried = hpc.submit(root, retry=True, runner=runner)["submissions"]
        self.assertEqual([receipt["task_indices"] for receipt in retried], [[1], [3]])
        self.assertTrue(all("--array=1%2" in receipt["command"] for receipt in retried))
        final = hpc.status(root, runner=runner)
        self.assertEqual([row["scheduler_job_id"] for row in final["tasks"]], ["100", "103", "101", "104", "102"])
        self.assertEqual([row["result_status"] for row in final["tasks"]], ["COMPLETE", "MISSING", "COMPLETE", "MISSING", "COMPLETE"])

    @unittest.skipUnless(os.name == "posix", "Slurm scripts require Bash")
    def test_second_shard_passes_its_global_index_to_the_worker(self):
        shell_probe = self.root / "python probe"
        shell_probe.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n', encoding="utf-8")
        shell_probe.chmod(0o700)
        root, _ = self._render(max_array_size=1, python=str(shell_probe))
        run = subprocess.run(["bash", str(root / "job-0002.sbatch")], env={**os.environ, "SLURM_ARRAY_TASK_ID": "0"},
                             capture_output=True, text=True, check=True)
        self.assertEqual(run.stdout.splitlines(), ["-m", "emulateqpu.execution", "--run-task", str(root.resolve()), "1"])

    def test_unknown_shard_submission_is_recorded_and_blocks_repeat_dispatch(self):
        root, _ = self._render(max_array_size=1)
        def runner(command, **kwargs):
            if command[-1].endswith("job.sbatch"):
                return subprocess.CompletedProcess(command, 0, "1234\n", "")
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        with self.assertRaisesRegex(RuntimeError, "shard 2"):
            hpc.submit(root, runner=runner)
        self.assertEqual(ex._read(root / "submission_0001.json")["job_id"], "1234")
        self.assertEqual(ex._read(root / "submission_0002.json")["status"], "SUBMISSION_ID_UNKNOWN")
        with patch("subprocess.run") as scheduler:
            with self.assertRaisesRegex(ValueError, "no confirmed job identity"):
                hpc.submit(root, retry=True)
            scheduler.assert_not_called()

    def test_legacy_array_and_receipts_keep_stable_retry_mapping(self):
        root, metadata = self._render()
        metadata.pop("shards")
        ex.atomic_json(root / "slurm.json", metadata)
        def runner(command, **kwargs):
            if command[0] == "sacct":
                first_job = "--jobs=41" in command
                output = "41_0|COMPLETED|0:0\n41_1|FAILED|1:0\n" if first_job else "42_1|COMPLETED|0:0\n"
            elif command[0] == "squeue":
                output = ""
            else:
                output = "42\n" if "--array=1%2" in command else "41\n"
            return subprocess.CompletedProcess(command, 0, output, "")
        for attempt in (1, 2):
            if attempt == 2:
                self._success(root, 0)
            receipt = hpc.submit(root, runner=runner, retry=(attempt == 2))
            receipt.pop("shard")
            receipt.pop("task_indices")
            if attempt == 1:
                receipt["command"] = [arg for arg in receipt["command"] if not arg.startswith("--array=")]
            ex.atomic_json(root / f"submission_{attempt:04d}.json", receipt)
        state = hpc.status(root, runner=runner)
        self.assertEqual([row["scheduler_job_id"] for row in state["tasks"]], ["41", "42"])

    def test_array_limit_and_scheduler_environment_are_validated(self):
        profile = {"shared_filesystem": True}
        self.assertEqual(hpc.validate_profile(profile)["max_array_size"], 1001)
        with self.assertRaisesRegex(ValueError, "max_array_size"):
            hpc.validate_profile({**profile, "max_array_size": 0})
        with self.assertRaisesRegex(ValueError, "override Slurm"):
            hpc.validate_profile({**profile, "environment": {"SLURM_ARRAY_TASK_ID": "0"}})

    def test_changed_script_and_directive_injection_rejected(self):
        root, _ = self._render()
        with (root / "job.sbatch").open("a") as stream: stream.write("echo changed\n")
        with self.assertRaisesRegex(ValueError, "script changed"):
            hpc.submit(root)
        with self.assertRaises(ValueError): hpc.validate_profile({"shared_filesystem": True, "account": "a\n#SBATCH --x"})
        with self.assertRaises(ValueError): hpc._directive("$WORK/run")


class PalaceTests(unittest.TestCase):
    def test_missing_executable_is_a_capability_result(self):
        with patch("shutil.which", return_value=None):
            self.assertFalse(palace.detect_palace("not-installed")["available"])

    def test_wrapper_plan_uses_one_launcher_and_explicit_build_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"; config.write_text('{"Solver":{"Device":"CPU"}}')
            executable = root / "palace"; executable.write_text("wrapper")
            identity = {"available": True, "executable": str(executable),
                        "executable_sha256": palace._sha(executable), "kind": "mpi_wrapper",
                        "wrapper_supports_threads": True, "wrapper_supports_launcher": True}
            with patch.object(palace, "detect_palace", return_value=identity):
                plan = palace.plan_palace(config, mpi_ranks=2)
                self.assertEqual(plan["command"], [str(executable), "-np", "2", "-nt", "1", str(config.resolve())])
                self.assertFalse(plan["field_solve_executed"])
                with self.assertRaisesRegex(ValueError, "OpenMP"):
                    palace.plan_palace(config, threads=2)
                config.write_text('{"Solver":{"Device":"GPU"}}')
                with self.assertRaisesRegex(ValueError, "CUDA/HIP"):
                    palace.plan_palace(config)


class NumericalExecutionTests(unittest.TestCase):
    def test_four_workers_and_single_worker_use_separate_budget_scopes(self):
        from emulateqpu.cli import main
        from emulateqpu.contracts import example_request

        request = example_request("small-pulse")
        study = {"schema": ex.STUDY_SCHEMA, "tasks": [
            {"id": f"worker-{index}", "request": deepcopy(request)} for index in range(4)]}
        four = ex.plan_study(study, workers=4)
        self.assertEqual(four["backend"], "local-process")
        self.assertIsNone(four["aggregate_memory_limit_bytes"])
        self.assertGreater(four["aggregate_estimated_peak_bytes"], 512 * 1024**2)
        with self.assertRaisesRegex(ValueError, "explicit study budget"):
            ex.plan_study(study, workers=4, memory_limit_bytes=512 * 1024**2)
        with self.assertRaisesRegex(ValueError, "explicit profile budget"):
            ex.plan_study(study, workers=4, profile={"aggregate_memory_mb": 512})
        with self.assertRaisesRegex(ValueError, "serial backend"):
            ex.plan_study(study, backend="serial", workers=4)
        request["resources"]["memory_mb"] = 140
        single = ex.plan_study(request)
        self.assertGreater(single["aggregate_estimated_peak_bytes"], 140 * 1024**2)
        self.assertLessEqual(single["per_task_estimated_peak_bytes"][0], 140 * 1024**2)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ex.atomic_json(root / "study.json", study)
            console = StringIO()
            with redirect_stdout(console):
                code = main(["run", str(root / "study.json"), "--workers", "4", "--output", str(root / "worker results")])
            self.assertEqual(code, 0, console.getvalue())
            self.assertEqual(json.loads(console.getvalue())["complete_tasks"], 4)

    def test_invalid_example_leaves_no_output_directory(self):
        from emulateqpu.cli import main

        with tempfile.TemporaryDirectory() as temporary, redirect_stderr(StringIO()):
            output = Path(temporary) / "parent" / "bad-example"
            self.assertEqual(main(["init", "--example", "typo", "--output", str(output)]), 2)
            self.assertFalse(output.parent.exists())

    def _interrupted_at_boundary(self, root):
        from emulateqpu.api import _run_request
        from emulateqpu.contracts import example_request

        run = ex.prepare_run(example_request("feedback"), root / "run")
        _, manifest, study, _ = ex._load_run(run)
        identity = manifest["tasks"][0]
        request = study["tasks"][0]["request"]
        attempt = run / "tasks" / identity["id"] / "attempt_0001"
        attempt.mkdir()
        ex.atomic_json(attempt / "request.json", request)
        # Use the worker's cooperative deadline and ensure one operation crosses it.
        with patch("emulateqpu.api.time") as clock:
            clock.monotonic.side_effect = (0., 1., 2.)
            with self.assertRaisesRegex(ex.WallTimeLimitError, "checkpoint preserved"):
                _run_request(request, attempt / "output", deadline=2.)
        self.assertTrue((attempt / "output/checkpoint.json").is_file())
        ex.atomic_json(attempt / "attempt.json", {**identity, "task_id": identity["id"], "status": "FAILED"})
        return run, request, attempt

    def test_default_resume_picks_up_published_operation_checkpoint(self):
        import numpy as np
        from emulateqpu.api import load_result, run_request

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, request, first = self._interrupted_at_boundary(root)
            before = ex._file_manifest(first)
            full = run_request(request, root / "full")
            summary = ex.resume_run(run)
            self.assertTrue(summary["complete"], summary)
            retry = first.with_name("attempt_0002")
            self.assertEqual(ex._read(retry / "attempt.json")["resumed_from_attempt"], first.name)
            resumed, resumed_branches = load_result(retry / "output")
            _, full_branches = load_result(root / "full")
            self.assertTrue(resumed["resumed_from_checkpoint"])
            self.assertEqual(resumed["duration_ns"], full["duration_ns"])
            self.assertEqual(resumed["summary"]["measurement_records"], full["summary"]["measurement_records"])
            for actual, expected in zip(resumed_branches, full_branches, strict=True):
                np.testing.assert_allclose(actual.density(), expected.density(), atol=1e-10, rtol=1e-10)
                np.testing.assert_array_equal(actual.frame, expected.frame)
            self.assertEqual(ex._file_manifest(first), before)

    def test_corrupt_checkpoint_fails_without_restarting(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, _, first = self._interrupted_at_boundary(Path(temporary))
            payload = next((first / "output/checkpoints").glob("*.npz"))
            with payload.open("ab") as stream:
                stream.write(b"corrupt")
            summary = ex.resume_run(run)
            self.assertFalse(summary["complete"])
            attempt = summary["tasks"][0]["attempts"][-1]
            self.assertIn("Checkpoint payload integrity mismatch", attempt["error"])
            self.assertFalse((first.with_name("attempt_0002") / "output/result.json").exists())

    def test_serial_process_and_resumed_study_agree(self):
        import numpy as np
        from emulateqpu.contracts import example_request
        from emulateqpu.io import inspect_result
        request = example_request("small-pulse")
        request["schedule"]["duration_ns"] = .04
        for drive in request["schedule"]["drives"]:
            drive["duration_ns"] = .04
        study = {"schema": ex.STUDY_SCHEMA, "tasks": [
            {"id": "first", "request": deepcopy(request)}, {"id": "second", "request": deepcopy(request)}]}
        with tempfile.TemporaryDirectory(prefix="emulateqpu-real-workers-") as temporary:
            root = Path(temporary)
            serial = ex.run_study(study, root / "serial")
            parallel = ex.run_study(study, root / "process", backend="local-process", workers=2)
            self.assertTrue(serial["complete"], serial)
            self.assertTrue(parallel["complete"], parallel)
            partial = ex.prepare_run(study, root / "resume")
            ex.execute_task(partial, 0)
            preserved = (partial / "tasks/first/attempt_0001/attempt.json").read_bytes()
            resumed = ex.resume_run(partial)
            self.assertTrue(resumed["complete"], resumed)
            self.assertEqual(preserved, (partial / "tasks/first/attempt_0001/attempt.json").read_bytes())
            for task in ("first", "second"):
                relative = Path("tasks") / task / "attempt_0001/output"
                expected = root / "serial" / relative
                for backend in ("process", "resume"):
                    actual = root / backend / relative
                    self.assertEqual(inspect_result(actual)["request_sha256"], inspect_result(expected)["request_sha256"])
                    with np.load(expected / "states.npz", allow_pickle=False) as a, np.load(actual / "states.npz", allow_pickle=False) as b:
                        self.assertEqual(set(a.files), set(b.files))
                        for key in a.files:
                            np.testing.assert_allclose(a[key], b[key], atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
