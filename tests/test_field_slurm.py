"""Field allocations and output provenance using a deterministic fake scheduler."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from emulateqpu import field_slurm
from emulateqpu.cli import main
from emulateqpu.io import atomic_json, file_hash, read_json


class FieldSlurmTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="field slurm ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.mesh = self.root / "mesh.msh"
        self.mesh.write_bytes(b"frozen test mesh\n")
        self.binary = self.root / "palace.bin"
        self.binary.write_bytes(b"\x7fELF mock field solver")
        self.build = self.root / "build.json"
        atomic_json(self.build, {"executable_sha256": file_hash(self.binary)})
        self.config = self.root / "case.json"
        atomic_json(self.config, {
            "Problem": {"Type": "Eigenmode", "Output": "original"},
            "Model": {"Mesh": self.mesh.name},
            "Solver": {"Eigenmode": {"N": 2}},
        })
        self.profile = {
            "schema": field_slurm.PROFILE_SCHEMA, "shared_filesystem": True,
            "shared_root": str(self.root), "python": sys.executable,
            "palace_binary": str(self.binary), "build_manifest": str(self.build),
            "mpi_ranks": 2, "mpi_type": "pmix", "partition": "cpu", "account": "pilot",
            "memory_mb": 1024, "wall_time": "00:05:00",
        }
        self.run_path = self.root / "field job"

    def stage(self):
        expected = {"domain-E.csv": 2}
        plan = field_slurm.plan(self.config, self.profile, required_outputs=expected)
        return field_slurm.render(plan, self.run_path)

    def queue(self):
        def fake(command, **kwargs):
            self.assertEqual(command, ["sbatch", "--parsable", str(self.run_path / "job.sbatch")])
            return subprocess.CompletedProcess(command, 0, "91;testcluster\n", "")
        return field_slurm.submit(self.run_path, runner=fake)

    def account(self, ranks=2):
        def fake(command, **kwargs):
            self.assertIn("--clusters=testcluster", command)
            lines = ("91|COMPLETED|0:0||13|\n91.batch|COMPLETED|0:0|1|13|31M\n"
                     f"91.0|COMPLETED|0:0|{ranks}|12|104M\n")
            return subprocess.CompletedProcess(command, 0, lines if command[0] == "sacct" else "", "")
        return fake

    def solve(self, table="m,Re{f} (GHz),Q\n1,5.1,inf\n2,7.2,5000\n", *, actual_ranks=2):
        def fake(command, **kwargs):
            self.assertEqual(command, ["srun", "--nodes=1", "--ntasks=2", "--cpus-per-task=1",
                                       "--kill-on-bad-exit=1", "--mpi=pmix", str(self.binary),
                                       str(self.run_path / "palace.json")])
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
            self.assertEqual(kwargs["cwd"], self.root)
            raw = (self.run_path / "raw")
            raw.mkdir()
            atomic_json(raw / "palace.json", {"Problem": {"MPISize": actual_ranks}})
            (raw / "eig.csv").write_text(table)
            (raw / "domain-E.csv").write_text("m,E_elec (J)\n1,1.0e-4\n2,2.0e-4\n")
            return subprocess.CompletedProcess(command, 0)
        env = {"SLURM_JOB_ID": "91", "SLURM_NTASKS": "2", "OMP_NUM_THREADS": "9"}
        return field_slurm._execute(self.run_path, runner=fake, environment=env)

    def test_plan_and_render_freeze_mesh_and_create_a_single_mpi_step(self):
        config_bytes = self.config.read_bytes()
        record = self.stage()
        self.assertEqual(self.config.read_bytes(), config_bytes)
        self.assertEqual(record["status"], "RENDERED")
        value = read_json(self.run_path / "palace.json")
        self.assertEqual(value["Model"]["Mesh"], str(self.mesh))
        self.assertEqual(value["Problem"]["Output"], str(self.run_path / "raw"))
        script = (self.run_path / "job.sbatch").read_text()
        self.assertIn("#SBATCH --nodes=1\n#SBATCH --ntasks=2\n#SBATCH --cpus-per-task=1", script)
        self.assertIn("#SBATCH --mem=1024M", script)
        self.assertIn(f"'{self.run_path}'", script)
        self.assertEqual(record["command"].count("srun"), 1)
        self.assertEqual(read_json(self.run_path / "plan.json")["profile"]["python"], sys.executable)

    @unittest.skipIf(sys.platform == "win32", "Field Slurm CLI requires POSIX")
    def test_cli_plan_render_and_collect_use_separate_paths(self):
        profile = self.root / "site.json"
        atomic_json(profile, self.profile)
        output = self.root / "field-plan.json"
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(main(["field", "plan", str(self.config), "--profile", str(profile),
                                   "--require-table", "port-EPR.csv=2", "--output", str(output)]), 0)
            self.assertEqual(main(["field", "render", str(output), "--output", str(self.run_path)]), 0)
            self.assertEqual(main(["field", "collect", str(self.run_path)]), 2)
        self.assertEqual(read_json(output)["required_outputs"], {"eig.csv": 2, "port-EPR.csv": 2})
        self.assertFalse((self.run_path / "submission.json").exists())

    def test_driven_plan_counts_explicit_points_and_accepts_zero_transmission(self):
        settings = read_json(self.config)
        settings["Problem"]["Type"] = "Driven"
        settings["Solver"] = {"Driven": {"Samples": [{"Type": "Point", "Freq": [4.2, 4.3]}]}}
        atomic_json(self.config, settings)
        value = field_slurm.plan(self.config, self.profile)
        self.assertEqual(value["required_outputs"], {"port-S.csv": 2})
        spectrum = self.root / "port-S.csv"
        spectrum.write_text("f (GHz),|S[1][1]| (dB),arg(S[1][1]) (deg.)\n4.2,-inf,0\n4.3,-45,15\n")
        field_slurm._check_csv(spectrum, 2)
        spectrum.write_bytes(spectrum.read_bytes().replace(b"-45", b"nan"))
        with self.assertRaisesRegex(ValueError, "Invalid numeric"):
            field_slurm._check_csv(spectrum, 2)

    def test_electrostatic_job_collects_declared_terminal_matrix(self):
        settings = read_json(self.config)
        settings["Problem"]["Type"] = "Electrostatic"
        settings["Boundaries"] = {"Terminal": [{"Index": 1, "Attributes": [9]},
                                               {"Index": 2, "Attributes": [10]}]}
        settings["Solver"] = {"Order": 2}
        atomic_json(self.config, settings)
        value = field_slurm.plan(self.config, self.profile)
        self.assertEqual(value["required_outputs"], {"terminal-C.csv": 2})
        field_slurm.render(value, self.run_path)
        self.queue()

        def fake(command, **kwargs):
            raw = self.run_path / "raw"
            raw.mkdir()
            atomic_json(raw / "palace.json", {"Problem": {"MPISize": 2}})
            (raw / "terminal-C.csv").write_text("i,C[i][1] (F),C[i][2] (F)\n1,8e-14,-5e-15\n2,-5e-15,7e-14\n")
            return subprocess.CompletedProcess(command, 0)

        env = {"SLURM_JOB_ID": "91", "SLURM_NTASKS": "2"}
        self.assertEqual(field_slurm._execute(self.run_path, runner=fake, environment=env)["status"], "COMPLETE")
        self.assertTrue(field_slurm.status(self.run_path, runner=self.account())["complete"])

        matrix = self.run_path / "raw" / "terminal-C.csv"
        matrix.write_text("i,C[i][1] (F)\n1,8e-14\n2,-5e-15\n")
        result = field_slurm.collect(self.run_path)
        self.assertEqual(result["status"], "CORRUPT")
        self.assertIn("expected terminal columns", result["error"])

    def test_scheduler_success_requires_result_and_expected_mpi_count(self):
        self.stage()
        self.queue()
        self.assertFalse(field_slurm.status(self.run_path, runner=self.account())["complete"])
        self.assertEqual(self.solve()["status"], "COMPLETE")
        self.assertTrue(field_slurm.collect(self.run_path)["complete"])
        self.assertFalse(field_slurm.status(self.run_path, runner=self.account(1))["complete"])
        status = field_slurm.status(self.run_path, runner=self.account())
        self.assertTrue(status["complete"])
        self.assertEqual(status["scheduler_steps"][0]["max_task_rss"], "104M")
        self.assertEqual(status["result"]["execution"]["actual_mpi_ranks"], 2)
        self.assertEqual(status["result"]["field_numerical_qualification"], "NOT_RUN")

    def test_successful_scheduler_does_not_hide_wrong_palace_mpi_size(self):
        self.stage()
        self.queue()
        result = self.solve(actual_ranks=1)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("reported MPI size 1, expected 2", result["error"])
        status = field_slurm.status(self.run_path, runner=self.account())
        self.assertTrue(status["ranks_confirmed"])
        self.assertFalse(status["complete"])

    def test_changed_palace_mpi_size_fails_collection(self):
        self.stage()
        self.assertEqual(self.solve()["status"], "COMPLETE")
        atomic_json(self.run_path / "raw" / "palace.json", {"Problem": {"MPISize": 1}})
        result = field_slurm.collect(self.run_path)
        self.assertEqual(result["status"], "CORRUPT")
        self.assertIn("reported MPI size 1, expected 2", result["error"])

    def test_truncated_or_invalid_csv_fails_even_with_successful_solver_exit(self):
        self.stage()
        result = self.solve("m,Re{f} (GHz)\n1,5.1\n")
        self.assertEqual(result["returncode"], 0)
        self.assertIn("expected at least 2", result["error"])
        self.assertFalse(field_slurm.collect(self.run_path)["complete"])

    def test_output_modification_invalidates_the_receipt(self):
        self.stage()
        self.assertEqual(self.solve()["status"], "COMPLETE")
        result = self.run_path / "raw" / "eig.csv"
        result.write_bytes(result.read_bytes().replace(b"7.2", b"7.3"))
        self.assertEqual(field_slurm.collect(self.run_path)["status"], "CORRUPT")

    def test_duplicate_submission_and_changed_input_rejected(self):
        self.stage()
        self.queue()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.queue()
        self.mesh.write_bytes(b"modified after planning")
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.solve()

    def test_unknown_submissions_are_preserved(self):
        self.stage()
        def fake(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        with self.assertRaisesRegex(RuntimeError, "identity was not established"):
            field_slurm.submit(self.run_path, runner=fake)
        self.assertEqual(read_json(self.run_path / "submission.json")["status"], "SUBMISSION_UNKNOWN")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.queue()

    def test_rejects_wrappers_and_nested_mpi(self):
        self.stage()
        with self.assertRaisesRegex(ValueError, "outside any MPI rank"):
            field_slurm._execute(self.run_path, environment={
                "SLURM_JOB_ID": "91", "SLURM_NTASKS": "2", "PMI_SIZE": "2",
            })
        self.binary.write_text("#!/bin/sh\n")
        atomic_json(self.build, {"executable_sha256": file_hash(self.binary)})
        with self.assertRaisesRegex(ValueError, "direct Linux Palace"):
            field_slurm.plan(self.config, self.profile)


if __name__ == "__main__":
    unittest.main()
