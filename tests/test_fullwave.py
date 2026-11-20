"""Numerical interface checks for the four-data full-wave workflow."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from emulateqpu.fullwave import design, reduction, response
from emulateqpu.io import atomic_json, digest, file_hash, read_json
from emulateqpu._engine.model import validate_cell


def _table(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


@unittest.skipUnless(importlib.util.find_spec("shapely"), "Physical profile required")
class FullwaveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.design, _ = design.four_qubit_design()
        self.train = [3., 5., 7., 9.]
        self.holdout = [4., 6., 8.]

    def _synthetic_case(self, refinement, *, holdout_gain=.5, field_samples=False):
        case = self.root / refinement
        case.mkdir()
        atomic_json(case / "design.json", self.design)
        (case / "mesh.msh").write_text(refinement)
        surfaces = {name: index + 1 for index, name in enumerate(["enclosure", *self.design["sheets_wkt"]])}
        modes = np.arange(1, 9)
        cfg = design._configs(self.design, surfaces, {"substrate": 501, "air": 502}, len(modes), self.train, self.holdout)
        for name, value in cfg.items():
            atomic_json(case / (name + ".json"), value)
        n_cells = 1 if refinement == "coarse" else 2
        atomic_json(case / "field-inputs.json", {"schema": "emulateqpu.fullwave.inputs.v1",
                                                "refinement": refinement, "tetrahedra": n_cells,
                                                "design_sha256": digest(self.design),
                                                "files_sha256": {path.name: file_hash(path) for path in case.iterdir()},
                                                "palace": {"tag": design.PALACE_TAG, "commit": design.PALACE_COMMIT}})
        for name in cfg:
            path = case / (name + "-output")
            path.mkdir()
            atomic_json(path / "palace.json", {"GitTag": design.PALACE_TAG,
                                              "Problem": {"MPISize": 1, "MeshElements": n_cells}})
        eigen = case / "eigenmode-output"
        solved = eigen / "gridfunction" / "eigenmode"
        solved.mkdir(parents=True)
        (solved / "mesh.000000").write_text(refinement)
        if field_samples:
            # Explicitly synthetic contract tests for the raw-field path.
            for mode in modes:
                for component in ("real", "imag"):
                    (solved / f"E_{component}_{mode:06d}.gf.000000").write_text("Synthetic test vectors")
        f = [4 + .4 * i + (0.0001 if refinement == "fine" else 0) for i in modes]
        imag = .0003
        _table(eigen / "eig.csv", ["m", "Re{f} (GHz)", "Im{f} (GHz)", "Error (Bkwd.)", "Q"],
               [[i, k, imag, 1e-9, np.hypot(k, imag) / (2 * imag)] for i, k in zip(modes, f)])
        elements = [junction["port"] for junction in self.design["junctions"]]
        ports = [port["port"] for port in self.design["ports"]]
        _table(eigen / "port-EPR.csv", ["m", *[f"p[{j}]" for j in elements]],
               [[i, *[(-.8 if i - 1 == j else .005) for j in range(len(elements))]] for i in modes])
        _table(eigen / "domain-E.csv", ["m", "E_elec (J)", "E_mag (J)", "E_cap (J)", "E_ind (J)"],
               [[i, 1e-20, (1 - fraction) * 1e-20, 0., fraction * 1e-20]
                for i, fraction in zip(modes, [.83] * 7 + [.035])])
        _table(eigen / "port-Q.csv", ["m", *[f"κ_ext[{p}] (GHz)" for p in ports]],
               [[i, *[2 * imag / len(ports)] * len(ports)] for i in modes])
        islands = range(1, len(self.design["island_order"]) + 1)
        _table(case / "electrostatic-output" / "terminal-C.csv", ["i", *[f"C[i][{i}] (F)" for i in islands]],
               [[i, *[35e-15 if i == j else 0. for j in islands]] for i in islands])
        for label, grid, gain in (("driven-train", self.train, .5), ("driven-holdout", self.holdout, holdout_gain)):
            headers, data = ["f (GHz)"], []
            for p in ports:
                for q in ports:
                    headers.extend((f"|S[{p}][{q}]| (dB)", f"arg(S[{p}][{q}]) (deg.)"))
                    diagonal_or_readout = p == q or {p, q} == {5, 6}
                    data.extend((20 * np.log10(gain) if diagonal_or_readout else -np.inf, 15.))
            _table(case / (label + "-output") / "port-S.csv", headers, [[value, *data] for value in grid])
        parsed = reduction._eigen(eigen, self.design, synthetic=not field_samples)
        if field_samples:
            atomic_json(eigen / "execution.json", {"status": "COMPLETE"})
            atomic_json(eigen / "fullwave-execution.json", {
                "schema": reduction.LOCAL_RUN_SCHEMA, "stage": "eigenmode", "status": "COMPLETE",
                "config_sha256": file_hash(case / "eigenmode.json"), "mesh_sha256": file_hash(case / "mesh.msh"),
                "execution_sha256": file_hash(eigen / "execution.json"), "field_sha256": parsed["file_sha256"]})
        return case, parsed

    def _inputs(self, *, holdout_gain=.5, field_samples=False):
        coarse, a = self._synthetic_case("coarse", field_samples=field_samples)
        fine, b = self._synthetic_case("fine", holdout_gain=holdout_gain, field_samples=field_samples)
        overlap_path = self.root / "overlap.json"
        size = len(b["modes"])
        atomic_json(overlap_path, {"schema": reduction.OVERLAP_SCHEMA, "coarse_modes": a["modes"].tolist(),
                                   "fine_modes": b["modes"].tolist(), "overlap_abs": (.99 * np.eye(size)).tolist(),
                                   "coarse_field_sha256": a["file_sha256"], "fine_field_sha256": b["file_sha256"]})
        return coarse, fine, overlap_path, a, b

    def test_fresh_mesh_and_port_boundaries(self):
        try:
            import gmsh
        except (ImportError, OSError) as error:
            self.skipTest(f"Gmsh runtime unavailable: {error}")
        self.assertTrue(gmsh.__version__)
        path = self.root / "mesh"
        manifest = design.prepare_design(path)
        self.assertEqual(manifest["campaign_status"], "MESH_ONLY")
        self.assertGreater(manifest["tetrahedra"], 1000)
        self.assertEqual(file_hash(path / "mesh.msh"), manifest["files_sha256"]["mesh.msh"])
        static, driven = (read_json(path / name) for name in ("electrostatic.json", "driven-train.json"))
        self.assertEqual(len(static["Boundaries"]["Terminal"]), 14)
        ports = driven["Boundaries"]["LumpedPort"]
        self.assertEqual([p["Excitation"] for p in ports if "Excitation" in p], list(range(1, 7)))
        self.assertEqual(len([p for p in ports if "L" in p]), 7)
        with self.assertRaises(FileExistsError):
            design.prepare_design(path)

    def test_qualified_synthetic_responses_and_static_binding(self):
        coarse, fine, overlap, _, _ = self._inputs()
        artifact = self.root / "qualified"
        record = reduction.qualify_reduction(coarse, fine, overlap, artifact, synthetic=True)
        self.assertTrue(record["passed"], record)
        self.assertTrue(record["synthetic"])
        self.assertTrue(record["checks"]["junction_modes_covered"])
        cell = validate_cell(read_json(artifact / "electrostatic-cell.json"))
        self.assertEqual(cell["parameter_origin"], "assumed_circuit")
        self.assertFalse(cell["provenance"]["geometry_extracted"])
        self.assertEqual(cell["idle_flux_Phi0"], [j["bias_Phi0"] for j in self.design["junctions"]])
        model = response.load_model(artifact)
        self.assertLess(model["zero_point_phase_rad"][0][0], 0)
        occupations = [0] * len(model["mode_ids"])
        self.assertEqual(response.fock_energy_GHz(model, occupations), 0.)
        occupations[0] = 1
        self.assertEqual(response.fock_energy_GHz(model, occupations), model["f01_GHz"][0])
        occupations[0] = 2
        self.assertAlmostEqual(response.fock_energy_GHz(model, occupations),
                               2 * model["f01_GHz"][0] + model["kerr_GHz"][0][0])
        for label in design.PORT_ORDER[:5]:
            result = response.drive_response(model, label, 5., 20., 1e-5)
            self.assertTrue(result["synthetic"])
            self.assertGreater(result["mode_population_at_tone_off"][0], 0.)
        self.assertAlmostEqual(result["output_power_fraction"]["readout_out"], .25)
        with self.assertRaises(ValueError):
            response.drive_response(model, "drive_q0", 12., 20., 1e-5)
        (artifact / "model.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            response.load_model(artifact)

    def test_failed_holdout_keeps_report_and_withholds_model(self):
        coarse, fine, overlap, _, _ = self._inputs(holdout_gain=.25)
        artifact = self.root / "failed"
        record = reduction.qualify_reduction(coarse, fine, overlap, artifact, synthetic=True)
        self.assertFalse(record["passed"])
        self.assertFalse(record["checks"]["heldout_complex_rms"])
        self.assertTrue((artifact / "qualification.json").is_file())
        self.assertFalse((artifact / "model.json").exists())
        with self.assertRaisesRegex(ValueError, "passing full-wave artifact"):
            response.load_model(artifact)

    def test_mesh_and_raw_identity_changes_are_rejected(self):
        coarse, fine, overlap, _, _ = self._inputs()
        artifact = self.root / "bad"
        csv_path = fine / "eigenmode-output" / "domain-E.csv"
        csv_path.write_bytes(csv_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            reduction.qualify_reduction(coarse, fine, overlap, artifact, synthetic=True)
        (coarse / "mesh.msh").write_text("changed mesh")
        with self.assertRaisesRegex(ValueError, "input integrity mismatch"):
            reduction.qualify_reduction(coarse, fine, overlap, artifact, synthetic=True)
        self.assertFalse(artifact.exists())

    def test_native_result_requires_original_config_and_mesh_receipt(self):
        coarse, _, _, _, _ = self._inputs(field_samples=True)
        case, manifest, _ = reduction._case(coarse, "coarse")
        stage = reduction._stage_output(case, manifest, "eigenmode")
        receipt = read_json(stage / "fullwave-execution.json")
        receipt["config_sha256"] = "0" * 64
        atomic_json(stage / "fullwave-execution.json", receipt)
        with self.assertRaisesRegex(ValueError, "stale, incomplete or lacks a bound launch"):
            reduction._stage_output(case, manifest, "eigenmode")

    def test_driven_grid_must_cover_junction_modes(self):
        self.train = [5., 7., 9.]
        self.holdout = [6., 8.]
        coarse, fine, overlap, _, _ = self._inputs()
        artifact = self.root / "band-failed"
        record = reduction.qualify_reduction(coarse, fine, overlap, artifact, synthetic=True)
        self.assertFalse(record["checks"]["junction_modes_covered"])
        self.assertFalse((artifact / "model.json").exists())

    def test_complex_field_overlap_is_phase_invariant(self):
        coarse, fine, _, a, b = self._inputs(field_samples=True)
        sample_path = self.root / "vectors.npz"
        modes = len(b["modes"])
        fields = np.zeros((modes, modes, 3), complex)
        fields[np.arange(modes), np.arange(modes), 0] = 1j
        np.savez(sample_path, points_mm=np.arange(modes * 3).reshape(modes, 3), weights=np.ones(modes),
                 coarse_modes=a["modes"], fine_modes=b["modes"], coarse_E=fields, fine_E=fields * np.exp(.7j),
                 coarse_fields_sha256=digest(a["file_sha256"]), fine_fields_sha256=digest(b["file_sha256"]))
        result = reduction.write_mode_overlap(coarse, fine, sample_path, self.root / "sample-overlap.json")
        np.testing.assert_allclose(result["overlap_abs"], np.eye(modes), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
