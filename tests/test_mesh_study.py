"""Exercise mesh study contracts with synthetic matrices and launch receipts."""
import csv
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from emulateqpu.fullwave import design, mesh_study
from emulateqpu.io import atomic_json, file_hash, read_json


@unittest.skipUnless(importlib.util.find_spec("shapely"), "Physical profile required")
class MeshStudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.study = self.root / "study"
        self.binary = self.root / "palace"
        self.binary.write_text("Synthetic Palace executable identity", encoding="utf-8")

    @staticmethod
    def _mesh(record, sheets, path, level):
        path.write_text(level, encoding="utf-8")
        bridges = [bridge["label"] for bridge in record.get("ground_bridges", [])]
        groups = {name: number for number, name in enumerate(["enclosure", *sheets, *bridges], 1)}
        return groups, {"air": 501, "substrate": 514}, 1000 * (list(design.LADDER_MESHES).index(level) + 1)

    def _prepare(self, *, order=2, full_assembly=False, legacy=False):
        with patch.object(design, "_mesh_design", side_effect=self._mesh):
            return mesh_study.prepare_study(self.study, order=order, full_assembly=full_assembly, legacy=legacy)

    def _launch_plan(self, case, *, tag=design.PALACE_TAG, ranks=2):
        config = case / "electrostatic.json"
        executable_hash = file_hash(self.binary)
        return {"schema": "emulateqpu.palace_plan.v1", "config": str(config), "config_sha256": file_hash(config),
                "command": [str(self.binary), str(config)], "device": "CPU", "mpi_ranks": ranks, "threads_per_rank": 1,
                "executable_identity": {"kind": "direct_binary_or_unknown", "executable": str(self.binary),
                                        "executable_sha256": executable_hash, "sibling_binary_candidates": []},
                "supplied_build_manifest": {"executable_sha256": executable_hash, "source_commit": design.PALACE_COMMIT}
                if tag == design.PALACE_TAG else {}}

    def _write_launch(self, level, *, scale=1., tag=design.PALACE_TAG, ranks=2, log="", solved_delta=0, adapt=0):
        case = self.study / level
        raw = case / design.LADDER_OUTPUT_DIRECTORY
        raw.mkdir()
        plan = self._launch_plan(case, tag=tag, ranks=ranks)
        count = read_json(case / "field-inputs.json")["tetrahedra"]
        terminal_count = len(read_json(case / "design.json")["island_order"])
        atomic_json(raw / "palace.json", {"GitTag": tag, "Problem": {"MPISize": ranks, "MeshElements": count + solved_delta},
                                          "ElapsedTime": {"Counts": {"Adaptation": adapt}},
                                          "LinearSolver": {"TotalSolves": terminal_count}})
        atomic_json(raw / "launch_plan.json", plan)
        receipt = {"schema": "emulateqpu.palace_execution.v1", "returncode": 0, "status": "COMPLETE", "resource_stop": None,
                   "plan": plan, "command": plan["command"], "elapsed_s": 1., "observed_peak_rss_bytes": 1024}
        atomic_json(raw / "execution.json", receipt)
        (raw / "stdout.log").write_text(log, encoding="utf-8")
        (raw / "stderr.log").write_text("", encoding="utf-8")
        c = scale * (35e-15 * np.eye(terminal_count) - 5e-15 * np.eye(terminal_count, k=1)
                     - 5e-15 * np.eye(terminal_count, k=-1))
        with (raw / "terminal-C.csv").open("w", encoding="utf-8", newline="") as table:
            writer = csv.writer(table)
            writer.writerow(["i", *(f"C[i][{i}] (F)" for i in range(1, terminal_count + 1))])
            writer.writerows([[i, *values] for i, values in enumerate(c, 1)])
        return receipt

    def test_diagnostic_finds_unconverged_interval_and_checks_effective_mesh_count(self):
        self._prepare(order=1)
        first, second, third = design.LADDER_MESHES
        self._write_launch(first, tag="b676277", scale=1.2, solved_delta=30,
                           log="Added 30 elements in 1 iterations of local bisection for under-resolved interior boundaries\n")
        self._write_launch(second, tag="b676277", scale=1.01)
        self._write_launch(third, tag="b676277")
        report = mesh_study.analyze_study(self.study, self.root / "report", diagnostic_unbound=True)
        self.assertEqual(report["status"], "DIAGNOSTIC_NOT_CONVERGED")
        self.assertFalse(report["electrostatic_mesh_qualified"])
        self.assertEqual(report["levels"][0]["solved_tetrahedra"], 1030)
        self.assertFalse(report["intervals"][0]["passed"])
        self.assertTrue(report["intervals"][1]["passed"])
        self.assertEqual(len(report["intervals"][0]["differential_EC_nearest_neighbors"]), 6)
        self.assertAlmostEqual(report["intervals"][1]["metrics"]["terminal_C_relative_frobenius"], .01)

    def test_unverified_inconsistent_or_partial_records_cannot_pass(self):
        self._prepare()
        first, second, third = design.LADDER_MESHES
        self._write_launch(first)
        self._write_launch(second, ranks=1)
        self._write_launch(third, adapt=1)
        report = mesh_study.analyze_study(self.study, self.root / "unverified")
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual([row["status"] for row in report["levels"]], ["UNVERIFIED"] * 3)
        report = mesh_study.analyze_study(self.study, self.root / "inconsistent", diagnostic_unbound=True)
        self.assertEqual(report["status"], "INCONSISTENT_EXECUTION")
        self.assertEqual(report["levels"][2]["status"], "INVALID")

    def test_source_bound_p2_and_tampered_native_receipts(self):
        self._prepare(full_assembly=True)
        for level in design.LADDER_MESHES:
            case = self.study / level
            config = read_json(case / "electrostatic.json")
            self.assertEqual(config["Solver"]["PartialAssemblyOrder"], 3)
            with (patch.object(mesh_study, "plan_palace", return_value=self._launch_plan(case)),
                  patch.object(mesh_study, "launch_palace", side_effect=lambda *_: self._write_launch(level))):
                mesh_study.run_level(self.study, level, executable=self.binary)
        report = mesh_study.analyze_study(self.study, self.root / "qualifies")
        self.assertEqual(report["status"], "CONVERGED")
        self.assertTrue(report["electrostatic_mesh_qualified"])
        output = self.study / level / design.LADDER_OUTPUT_DIRECTORY / "terminal-C.csv"
        output.write_text("Tampered", encoding="utf-8")
        report = mesh_study.analyze_study(self.study, self.root / "tampered", diagnostic_unbound=True)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["levels"][-1]["status"], "INVALID")
        self.assertFalse(report["electrostatic_mesh_qualified"])

    def test_mesh_or_threshold_tampering_invalidates_study_before_analysis(self):
        self._prepare()
        case = self.study / next(iter(design.LADDER_MESHES))
        (case / "mesh.msh").write_text("Changed mesh", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Palace input integrity"):
            mesh_study.analyze_study(self.study, self.root / "report")

    def test_legacy_ladder_retains_its_original_mesh_cap(self):
        self._prepare(legacy=True)
        study, _ = mesh_study._study(self.study)
        self.assertEqual([level["mesh_settings"]["maximum_tetrahedra"] for level in study["levels"]],
                         [200_000] * len(design.LADDER_MESHES))

    def test_shared_ladder_cli_preserves_one_design_across_levels(self):
        frequencies = [6.40, 6.62, 6.84, 7.08]
        shared_design, _ = design.four_qubit_design(readout_topology="shared", drive_gap_um=40,
                                                    readout_frequencies_GHz=frequencies)
        with patch.object(design, "_mesh_design", side_effect=self._mesh), redirect_stdout(io.StringIO()):
            self.assertEqual(mesh_study.main(["prepare", str(self.study), "--readout-topology", "shared",
                                              "--drive-gap-um", "40", "--readout-frequencies-ghz",
                                              *map(str, frequencies)]), 0)
        for level in design.LADDER_MESHES:
            case = self.study / level
            self.assertEqual(read_json(case / "design.json"), shared_design)
            static = read_json(case / "electrostatic.json")["Boundaries"]
            self.assertEqual(len(static["Terminal"]), 14)
            surfaces = read_json(case / "field-inputs.json")["physical_surface_ids"]
            self.assertTrue(all(surfaces[bridge["label"]] in static["Ground"]["Attributes"]
                                for bridge in shared_design["ground_bridges"]))
        self.assertEqual(len(mesh_study._study(self.study)[0]["levels"]), 3)

    def test_legacy_and_shared_ladder_conflict_before_writing(self):
        with self.assertRaises(ValueError):
            mesh_study.prepare_study(self.study, legacy=True, readout_topology="shared")
        self.assertFalse(self.study.exists())
