"""Bounded independent-task and mock-scheduler acceptance checks."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
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
    def _render(self, path="slurm run"):
        profile = {"schema": hpc.PROFILE_SCHEMA, "shared_filesystem": True,
                   "shared_root": str(self.root), "memory_mb": 128, "wall_time": "00:01:00",
                   "array_concurrency": 2, "cpus_per_task": 1, "max_retries": 1,
                   "partition": "short", "account": "project", "environment": {"CASE": "a b"}}
        with patch("emulateqpu.contracts.validate_request", side_effect=deepcopy), \
             patch.object(ex, "_plan_validated_request", side_effect=_fixture_plan):
            metadata = hpc.render(self.study, profile, self.root / path)
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
