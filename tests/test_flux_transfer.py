"""Synthetic Palace receipts and signed static flux transfer matrices."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from emulateqpu.fullwave import flux_transfer, reduction
from emulateqpu.fullwave.assembly import FLUX_TABLES, prepare_assembly
from emulateqpu.fullwave.assembly_mesh import MATERIAL_IDS
from emulateqpu.fullwave.design import PALACE_TAG
from emulateqpu.io import atomic_json, file_hash, read_json


def _write_table(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _write_native_tables(directory, transfer, currents, source_indices=(1, 2, 3)):
    flux_Wb = transfer / flux_transfer.PHI0_PER_MA_PER_H * np.asarray(currents)
    ordinal = {index: step for step, index in enumerate(sorted(source_indices), 1)}
    _write_table(directory / "surface-F.csv", ["i", *(f"Φ_mag[{i}] (Wb)" for i in (1, 2, 3))],
                 [[ordinal[source_indices[i]], *flux_Wb[:, i]] for i in (2, 0, 1)])
    _write_table(directory / "terminal-I.csv", ["i", "I_inc[i] (A)"],
                 [[source_indices[i], currents[i]] for i in (1, 2, 0)])


class FluxTransferTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.matrix = np.array([[.230, -.002, 2e-5], [.004, -.192, 0.], [.001, .003, .201]])

    def test_native_reader_reorders_indices_and_uses_reported_source_currents(self):
        source_indices = (41, 7, 19)
        setup = {"sources": [{"index": i} for i in source_indices], "pickups": [{"index": i} for i in (1, 2, 3)]}
        currents = [.041, .0515, .062]
        _write_native_tables(self.root, self.matrix, currents, source_indices=source_indices)
        matrix_H, observed = flux_transfer._flux_matrix(self.root, setup)
        np.testing.assert_array_equal(observed, currents)
        np.testing.assert_allclose(matrix_H * flux_transfer.PHI0_PER_MA_PER_H, self.matrix, rtol=1e-12)
        _write_table(self.root / "terminal-I.csv", ["i", "I_inc[i] (A)"], [(1, .04), (1, .05), (3, .06)])
        with self.assertRaisesRegex(ValueError, "duplicate Palace indices"):
            flux_transfer._flux_matrix(self.root, setup)
        _write_native_tables(self.root, self.matrix, [.041, 0., .062], source_indices=source_indices)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            flux_transfer._flux_matrix(self.root, setup)

    def test_mesh_agreement_allows_low_sign_noise_but_rejects_resolved_reversal(self):
        fine = self.matrix / flux_transfer.PHI0_PER_MA_PER_H
        coarse = fine * 1.02
        coarse[0, 2] = -fine[0, 2]
        self.assertTrue(flux_transfer._mesh_agreement(coarse, fine)["passed"])
        coarse[0, 1] = -fine[0, 1]
        result = flux_transfer._mesh_agreement(coarse, fine)
        self.assertFalse(result["passed"])
        self.assertFalse(result["entry_passed"][0][1])
        self.assertTrue(result["entry_passed"][0][2])

    @unittest.skipUnless(importlib.util.find_spec("shapely"), "Physical profile required")
    def test_reduction_binds_matched_cases_and_reports_fine_signed_units(self):
        def mesh(path, *, planar, nonplanar, refinement, **_):
            path.write_text(f"Synthetic {refinement} mesh\n", encoding="utf-8")
            groups = {name: index for index, name in enumerate(("enclosure", *planar, *nonplanar), 1)}
            return groups, MATERIAL_IDS, 1000 if refinement == "coarse" else 2000

        for refinement, transfer in (("coarse", self.matrix * 1.01), ("fine", self.matrix)):
            case = self.root / refinement
            with patch("emulateqpu.fullwave.assembly_mesh.mesh_assembly", side_effect=mesh):
                manifest = prepare_assembly(case, refinement=refinement, flux_transfer=True)
            native = case / "magnetostatic-output"
            native.mkdir()
            atomic_json(native / "palace.json", {"GitTag": PALACE_TAG, "Problem": {"MPISize": 1,
                        "MeshElements": manifest["tetrahedra"]}, "ElapsedTime": {"Counts": {"Adaptation": 0}}})
            (native / "stdout.log").write_text("Synthetic solver output\n", encoding="utf-8")
            (native / "stderr.log").write_text("", encoding="utf-8")
            _write_native_tables(native, transfer, [.041, .0515, .062])
            atomic_json(native / "execution.json", {"status": "COMPLETE"})
            atomic_json(native / "fullwave-execution.json", {"schema": reduction.LOCAL_RUN_SCHEMA,
                        "stage": "magnetostatic", "status": "COMPLETE",
                        "config_sha256": manifest["files_sha256"]["magnetostatic.json"],
                        "mesh_sha256": manifest["files_sha256"]["mesh.msh"],
                        "execution_sha256": file_hash(native / "execution.json"),
                        "field_sha256": reduction._field_provenance(native, FLUX_TABLES, solved_mesh=False, synthetic=False),
                        "mesh_preprocessing": reduction._mesh_preprocessing(native, manifest["tetrahedra"])})

        coarse, fine, output = (self.root / name for name in ("coarse", "fine", "record"))
        report = flux_transfer.reduce_flux_transfer(coarse, fine, output)
        self.assertEqual(report, read_json(output / "flux-transfer.json"))
        self.assertEqual(report["status"], "MESH_AGREEMENT")
        self.assertEqual(report["loop_order"], ["c01", "c12", "c23"])
        self.assertEqual(report["line_order"], ["flux_c01", "flux_c12", "flux_c23"])
        np.testing.assert_allclose(report["applied_flux_per_current_Phi0_per_mA"], self.matrix, rtol=1e-12)
        self.assertFalse(report["physical_device_calibrated"])
        native = fine / "magnetostatic-output"
        self.assertEqual(report["provenance"]["fine"]["native_output_sha256"]["terminal-I.csv"],
                         file_hash(native / "terminal-I.csv"))
        surface = native / "surface-F.csv"
        original = surface.read_bytes()
        surface.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "stale, incomplete or lacks a bound launch"):
            flux_transfer.reduce_flux_transfer(coarse, fine, self.root / "tampered")
        surface.write_bytes(original)
        stdout = native / "stdout.log"
        stdout.write_text("Linear solver did not converge, norm(Ax-b)/norm(b) = 1.724e-03!\n", encoding="utf-8")
        receipt_path = native / "fullwave-execution.json"
        receipt = read_json(receipt_path)
        receipt["mesh_preprocessing"]["log_sha256"]["stdout.log"] = file_hash(stdout)
        atomic_json(receipt_path, receipt)
        rejected = self.root / "nonconverged"
        with self.assertRaisesRegex(ValueError, "Palace reported a nonconverged solve"):
            flux_transfer.reduce_flux_transfer(coarse, fine, rejected)
        self.assertFalse(rejected.exists())
