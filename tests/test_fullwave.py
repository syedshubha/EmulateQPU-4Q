"""Numerical interface checks for the four-data full-wave workflow."""
import csv
import importlib.util
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.constants import elementary_charge, h

from emulateqpu.fullwave import design, reduction, response
from emulateqpu.fullwave.__main__ import main
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
        self.design, self.sheets = design.four_qubit_design()
        self.sources = {}
        self.train = [3., 5., 7., 9.]
        self.holdout = [4., 6., 8.]

    def _synthetic_case(self, refinement, *, holdout_gain=.5, field_samples=False, degraded_readout=None):
        case = self.root / refinement
        case.mkdir()
        atomic_json(case / "design.json", self.design)
        for name, source in self.sources.items():
            atomic_json(case / name, source)
        (case / "mesh.msh").write_text(refinement)
        surface_names = ["enclosure", *self.design["sheets_wkt"]]
        surface_names.extend(bridge["label"] for bridge in self.design.get("ground_bridges", ()))
        surface_names.extend(self.design.get("raised_sheets_mm", {}))
        surfaces = {name: index + 1 for index, name in enumerate(surface_names)}
        materials = {name: 500 + index for index, name in enumerate(
            self.design.get("material_permittivity", {"substrate": 0, "air": 0}), 1)}
        channels = self.design.get("readout_channels", [
            {"qubit": "q1", "input_port": "readout_in", "output_port": "readout_out"}])
        elements = [junction["port"] for junction in self.design["junctions"]]
        modes = np.arange(1, len(elements) + len(channels) + 1)
        cfg = design._configs(self.design, surfaces, materials, len(modes), self.train, self.holdout)
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
                                              "Problem": {"MPISize": 1, "MeshElements": n_cells},
                                              "ElapsedTime": {"Counts": {"Adaptation": 0}}})
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
        ports_by_label = {port["label"]: port["port"] for port in self.design["ports"]}
        ports = list(ports_by_label.values())
        readout_pairs = {row["qubit"]: frozenset((ports_by_label[row["input_port"]], ports_by_label[row["output_port"]]))
                         for row in channels}
        participation = np.full((len(modes), len(elements)), .005)
        for index, mode in enumerate(design.MODE_ORDER):
            group = [i for i, junction in enumerate(self.design["junctions"])
                     if junction.get("mode", junction["label"]) == mode]
            for offset, branch in enumerate(group):
                participation[index, branch] = (-1)**(offset + 1) * .8 / len(group)
        _table(eigen / "port-EPR.csv", ["m", *[f"p[{j}]" for j in elements]],
               [[i, *values] for i, values in zip(modes, participation, strict=True)])
        energy_fractions = np.abs(participation).sum(axis=1)
        _table(eigen / "domain-E.csv", ["m", "E_elec (J)", "E_mag (J)", "E_cap (J)", "E_ind (J)"],
               [[i, 1e-20, (1 - fraction) * 1e-20, 0., fraction * 1e-20]
                for i, fraction in zip(modes, energy_fractions)])
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
                    pair = frozenset((p, q))
                    transfer_gain = gain if p == q or pair in readout_pairs.values() else 0.
                    if label == "driven-holdout" and pair == readout_pairs.get(degraded_readout):
                        transfer_gain -= .03
                    data.extend((20 * np.log10(transfer_gain) if transfer_gain else -np.inf, 15.))
            _table(case / (label + "-output") / "port-S.csv", headers, [[value, *data] for value in grid])
        parsed = reduction._eigen(eigen, self.design, synthetic=not field_samples)
        if field_samples:
            atomic_json(eigen / "execution.json", {"status": "COMPLETE"})
            atomic_json(eigen / "fullwave-execution.json", {
                "schema": reduction.LOCAL_RUN_SCHEMA, "stage": "eigenmode", "status": "COMPLETE",
                "config_sha256": file_hash(case / "eigenmode.json"), "mesh_sha256": file_hash(case / "mesh.msh"),
                "execution_sha256": file_hash(eigen / "execution.json"), "field_sha256": parsed["file_sha256"]})
        return case, parsed

    def _inputs(self, *, holdout_gain=.5, field_samples=False, degraded_readout=None):
        coarse, a = self._synthetic_case("coarse", field_samples=field_samples)
        fine, b = self._synthetic_case("fine", holdout_gain=holdout_gain, field_samples=field_samples,
                                       degraded_readout=degraded_readout)
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
        excited = [p for p in ports if "Excitation" in p]
        self.assertEqual([p["Excitation"] for p in excited], list(range(1, 13)))
        expected_attributes = [[manifest["physical_surface_ids"][f"port_{label}"]] for label in design.PORT_ORDER]
        self.assertEqual([p["Attributes"] for p in excited], expected_attributes)
        self.assertEqual(len(set(row[0] for row in expected_attributes)), len(design.PORT_ORDER))
        self.assertEqual(len([p for p in ports if "L" in p]), 7)
        with self.assertRaises(FileExistsError):
            design.prepare_design(path)

    def test_four_readout_sheets_have_two_launches_and_a_qubit_pickup(self):
        cpw = self.design["readout_cpw"]
        islands = self.design["island_order"]
        for channel in self.design["readout_channels"]:
            with self.subTest(qubit=channel["qubit"]):
                resonator = self.sheets[channel["resonator"]]
                feedline = self.sheets[channel["feedline"]]
                self.assertAlmostEqual(resonator.distance(feedline), cpw["coupling_gap_mm"])
                for role in ("input_port", "output_port"):
                    launch = self.sheets[f"port_{channel[role]}"]
                    contact = launch.boundary.intersection(feedline.boundary)
                    self.assertAlmostEqual(contact.length, cpw["trace_width_mm"])
                nearest = min(islands, key=lambda name: resonator.distance(self.sheets[name]))
                qubit_index = self.design["mode_order"].index(channel["qubit"])
                self.assertEqual(nearest, islands[2 * qubit_index + 1])
                self.assertGreater(resonator.distance(self.sheets[nearest]), 0.)

    def test_qualified_synthetic_responses_and_static_binding(self):
        coarse, fine, overlap, _, _ = self._inputs()
        artifact = self.root / "qualified"
        record = reduction.qualify_reduction(coarse, fine, overlap, artifact, synthetic=True)
        self.assertTrue(record["passed"], record)
        self.assertTrue(record["synthetic"])
        self.assertTrue(record["checks"]["junction_modes_covered"])
        self.assertTrue(record["checks"][reduction.READOUT_HOLDOUT_METRIC])
        self.assertEqual([item["qubit"] for item in record["readout_through_interpolation"]],
                         ["q0", "q1", "q2", "q3"])
        cell = validate_cell(read_json(artifact / "electrostatic-cell.json"))
        self.assertEqual(cell["parameter_origin"], "assumed_circuit")
        self.assertFalse(cell["provenance"]["geometry_extracted"])
        self.assertEqual(cell["idle_flux_Phi0"], [j["bias_Phi0"] for j in self.design["junctions"]])
        model = response.load_model(artifact)
        self.assertEqual(model["readout_channels"], self.design["readout_channels"])
        self.assertLess(model["zero_point_phase_rad"][0][0], 0)
        occupations = [0] * len(model["mode_ids"])
        self.assertEqual(response.fock_energy_GHz(model, occupations), 0.)
        occupations[0] = 1
        self.assertEqual(response.fock_energy_GHz(model, occupations), model["f01_GHz"][0])
        occupations[0] = 2
        self.assertAlmostEqual(response.fock_energy_GHz(model, occupations),
                               2 * model["f01_GHz"][0] + model["kerr_GHz"][0][0])
        for label in design.PORT_ORDER:
            result = response.drive_response(model, label, 5., 20., 1e-5)
            self.assertTrue(result["synthetic"])
            self.assertGreater(result["mode_population_at_tone_off"][0], 0.)
            self.assertEqual(tuple(result["output_power_fraction"]), design.PORT_ORDER)
        for channel in model["readout_channels"]:
            result = response.drive_response(model, channel["input_port"], 5., 20., 1e-5)
            outputs = result["output_power_fraction"]
            self.assertAlmostEqual(outputs[channel["output_port"]], .25)
            self.assertTrue(all(outputs[other["output_port"]] == 0. for other in model["readout_channels"]
                                if other != channel))
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

    def test_degraded_readout_cannot_hide_in_matrix_average(self):
        root = self.root
        for topology in ("dedicated", "shared"):
            with self.subTest(topology=topology):
                self.design, _ = design.four_qubit_design(readout_topology=topology)
                self.root = root / topology
                self.root.mkdir()
                coarse, fine, overlap, _, _ = self._inputs(degraded_readout="q3")
                artifact = self.root / "readout-failed"
                record = reduction.qualify_reduction(coarse, fine, overlap, artifact, synthetic=True)
                self.assertTrue(record["checks"]["heldout_complex_rms"])
                self.assertFalse(record["checks"][reduction.READOUT_HOLDOUT_METRIC])
                errors = record["readout_through_interpolation"]
                self.assertAlmostEqual(errors[-1]["forward_heldout_complex_rms"], .03)
                self.assertAlmostEqual(errors[-1]["reverse_heldout_complex_rms"], .03)
                self.assertFalse((artifact / "model.json").exists())

    def test_shared_readout_qualifies_one_through_pair_for_four_resonators(self):
        self.design, _ = design.four_qubit_design(readout_topology="shared")
        coarse, fine, overlap, _, _ = self._inputs()
        artifact = self.root / "shared"
        record = reduction.qualify_reduction(coarse, fine, overlap, artifact, synthetic=True)
        self.assertTrue(record["passed"], record)
        self.assertTrue(record["checks"][reduction.READOUT_HOLDOUT_METRIC])
        (through,) = record["readout_through_interpolation"]
        self.assertEqual(through["qubits"], ["q0", "q1", "q2", "q3"])
        self.assertEqual((through["input_port"], through["output_port"]), ("readout_in", "readout_out"))
        model = response.load_model(artifact)
        self.assertEqual(model["readout_topology"], "shared")
        self.assertEqual(model["readout_channels"], self.design["readout_channels"])
        self.assertEqual(len({row["resonator"] for row in model["readout_channels"]}), 4)
        self.assertEqual(np.shape(model["scattering_real"]), (len(self.train), 6, 6))
        for source, destination in (("readout_in", "readout_out"), ("readout_out", "readout_in")):
            result = response.drive_response(model, source, 6.8, 20., 1e-5)
            self.assertEqual(tuple(result["output_power_fraction"]), design.SHARED_PORT_ORDER)
            self.assertAlmostEqual(result["output_power_fraction"][destination], .25)

    def test_assembly_reduction_groups_squid_branches_and_retains_full_carrier_scattering(self):
        from emulateqpu.fullwave.assembly import assembly_geometry
        from emulateqpu.fullwave.carrier_layout import reference_carrier_design
        from emulateqpu.fullwave.hardware import reference_hardware_design

        dedicated = reference_hardware_design()
        shared = reference_hardware_design(readout_topology="shared")
        carrier = reference_carrier_design(dedicated, shared)
        self.design, _, _, _ = assembly_geometry(shared, carrier)
        self.sources = {"chip-layout.json": shared, "carrier-layout.json": carrier}
        coarse, fine, overlap, _, _ = self._inputs()
        artifact = self.root / "assembly"
        record = reduction.qualify_reduction(coarse, fine, overlap, artifact, synthetic=True)
        self.assertTrue(record["passed"], record)
        self.assertEqual([assignment["junction"] for assignment in record["junction_mode_assignments"]],
                         list(design.MODE_ORDER))
        coupler = [row for row in self.design["junctions"] if row["mode"] == "c01"]
        assignment = record["junction_mode_assignments"][1]
        self.assertEqual(assignment["junction_branches"], [row["label"] for row in coupler])
        self.assertAlmostEqual(assignment["participation_abs"], .8)
        cell = validate_cell(read_json(artifact / "electrostatic-cell.json"))
        self.assertAlmostEqual(cell["EJ_idle_GHz"][1], sum(row["EJ_GHz"] for row in coupler))
        cj = sum(row["capacitance_F"] for row in coupler)
        self.assertAlmostEqual(cell["EC_matrix_GHz"][1][1], elementary_charge**2 / (h * 1e9 * (35e-15 + 2 * cj)))
        model = response.load_model(artifact)
        self.assertEqual(np.shape(model["zero_point_phase_rad"])[1], 10)
        self.assertLess(model["zero_point_phase_rad"][1][1], 0.)
        self.assertGreater(model["zero_point_phase_rad"][1][2], 0.)
        self.assertEqual(np.shape(model["scattering_real"]), (len(self.train), 15, 15))
        self.assertEqual(sum(port["active"] for port in model["ports"]), 9)
        driven = response.drive_response(model, "readout_q0_in", 6.8, 20., 1e-5)
        self.assertEqual(tuple(driven["output_power_fraction"]), design.ASSEMBLY_PORT_ORDER)
        self.assertAlmostEqual(driven["output_power_fraction"]["readout_q3_out"], .25)
        tampered = read_json(coarse / "chip-layout.json")
        tampered["scope"] = "Changed source record"
        atomic_json(coarse / "chip-layout.json", tampered)
        manifest = read_json(coarse / "field-inputs.json")
        manifest["files_sha256"]["chip-layout.json"] = file_hash(coarse / "chip-layout.json")
        atomic_json(coarse / "field-inputs.json", manifest)
        with self.assertRaisesRegex(ValueError, "source layout hashes differ"):
            reduction._case(coarse, "coarse")

    def test_local_mesh_bisection_is_bound_and_excludes_adaptivity(self):
        atomic_json(self.root / "palace.json", {"Problem": {"MeshElements": 110},
                                               "ElapsedTime": {"Counts": {"Adaptation": 0}}})
        stdout = self.root / "stdout.log"
        stdout.write_text("Added 10 elements in 2 iterations of local bisection for under-resolved interior boundaries\n")
        (self.root / "stderr.log").write_text("")
        receipt = reduction._mesh_preprocessing(self.root, 100)
        self.assertEqual(receipt["added_tetrahedra"], 10)
        self.assertEqual(receipt["log_sha256"]["stdout.log"], file_hash(stdout))
        atomic_json(self.root / "palace.json", {"Problem": {"MeshElements": 110},
                                               "ElapsedTime": {"Counts": {"Adaptation": 1}}})
        with self.assertRaisesRegex(ValueError, "verified geometric preprocessing"):
            reduction._mesh_preprocessing(self.root, 100)

    def test_bare_die_local_solver_binds_bisection_and_validates_receipt(self):
        coarse, _ = self._synthetic_case("coarse", field_samples=True)
        case, manifest, _ = reduction._case(coarse, "coarse")
        native = case / "eigenmode-output"
        config_hash = manifest["files_sha256"]["eigenmode.json"]
        record = read_json(native / "palace.json")
        record["Problem"]["MeshElements"] += 10
        atomic_json(native / "palace.json", record)
        log = native / "stdout.log"
        log.write_text("Added 10 elements in 2 iterations of local bisection for under-resolved interior boundaries\n")
        (native / "stderr.log").write_text("")
        plan = {"config_sha256": config_hash, "mpi_ranks": 1}
        with (patch("emulateqpu.palace.plan_palace", return_value=plan),
              patch("emulateqpu.palace.launch_palace", return_value={"status": "COMPLETE"})):
            receipt = reduction.solve_local(case, "eigenmode", executable="mock-palace")
        self.assertEqual(receipt["mesh_preprocessing"]["added_tetrahedra"], 10)
        self.assertEqual(reduction._stage_output(case, manifest, "eigenmode"), native)
        with log.open("a") as stream:
            stream.write("Unbound log edit\n")
        with self.assertRaisesRegex(ValueError, "mesh preprocessing logs differ"):
            reduction._stage_output(case, manifest, "eigenmode")
        del receipt["mesh_preprocessing"]
        atomic_json(native / "fullwave-execution.json", receipt)
        with self.assertRaisesRegex(ValueError, "solved mesh cell count differs"):
            reduction._stage_output(case, manifest, "eigenmode")

    def test_local_solve_cli_requires_reported_mpi_rank_count_to_match_plan(self):
        case, _ = self._synthetic_case("coarse")
        case = case.resolve()
        native = case / "electrostatic-output"
        config = case / "electrostatic.json"
        build = case / "build.json"
        plan = {"config_sha256": file_hash(config), "mpi_ranks": 2}
        execution = {"status": "COMPLETE", "plan": plan}
        atomic_json(native / "execution.json", execution)
        for name in ("stdout.log", "stderr.log"):
            (native / name).write_text("")
        args = ["solve", str(case), "electrostatic", "--executable", "mock-palace", "--mpi-ranks", "2",
                "--launcher", "mpirun", "--build-manifest", str(build)]
        errors = StringIO()
        with (patch("emulateqpu.palace.plan_palace", return_value=plan) as planner,
              patch("emulateqpu.palace.launch_palace", return_value=execution),
              redirect_stderr(errors), redirect_stdout(StringIO())):
            with self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Palace reported MPI size 1, expected 2", errors.getvalue())
            self.assertFalse((native / "fullwave-execution.json").exists())
            record = read_json(native / "palace.json")
            record["Problem"]["MPISize"] = 2
            atomic_json(native / "palace.json", record)
            self.assertEqual(main(args), 0)
        planner.assert_called_with(config, executable="mock-palace", mpi_ranks=2, launcher="mpirun",
                                   build_manifest=build, memory_mb=4096, wall_time_s=600)
        case, manifest, _ = reduction._case(case, "coarse")
        self.assertEqual(reduction._stage_output(case, manifest, "electrostatic"), native)
        execution["plan"]["mpi_ranks"] = 3
        execution_path = native / "execution.json"
        atomic_json(execution_path, execution)
        receipt_path = native / "fullwave-execution.json"
        receipt = read_json(receipt_path)
        receipt["execution_sha256"] = file_hash(execution_path)
        atomic_json(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "Palace reported MPI size 2, expected 3"):
            reduction._stage_output(case, manifest, "electrostatic")

    def test_legacy_readout_can_still_be_qualified(self):
        self.design, _ = design.four_qubit_design(legacy=True)
        coarse, fine, overlap, _, _ = self._inputs()
        artifact = self.root / "legacy"
        record = reduction.qualify_reduction(coarse, fine, overlap, artifact, synthetic=True)
        self.assertTrue(record["passed"], record)
        self.assertNotIn(reduction.READOUT_HOLDOUT_METRIC, record["checks"])
        model = response.load_model(artifact)
        result = response.drive_response(model, "readout_in", 5., 20., 1e-5)
        self.assertEqual(tuple(result["output_power_fraction"]), design.LEGACY_PORT_ORDER)
        self.assertAlmostEqual(result["output_power_fraction"]["readout_out"], .25)

    def _rebind_changed_design(self, case, changed):
        atomic_json(case / "design.json", changed)
        manifest = read_json(case / "field-inputs.json")
        manifest["design_sha256"] = digest(changed)
        manifest["files_sha256"]["design.json"] = file_hash(case / "design.json")
        atomic_json(case / "field-inputs.json", manifest)

    def test_changed_readout_mapping_is_rejected(self):
        coarse, _, _, _, _ = self._inputs()
        changed = read_json(coarse / "design.json")
        changed["readout_channels"][0]["output_port"] = "readout_q1_out"
        self._rebind_changed_design(coarse, changed)
        with self.assertRaisesRegex(ValueError, "Readout channel order or port mapping differs"):
            reduction._case(coarse, "coarse")

    def test_shared_case_rejects_a_second_feedline(self):
        self.design, _ = design.four_qubit_design(readout_topology="shared")
        coarse, _, _, _, _ = self._inputs()
        changed = read_json(coarse / "design.json")
        changed["readout_channels"][0]["feedline"] = "chip_ground"
        self._rebind_changed_design(coarse, changed)
        with self.assertRaisesRegex(ValueError, "Readout feedline mapping differs"):
            reduction._case(coarse, "coarse")

    def test_shared_case_rejects_swapped_resonators(self):
        self.design, _ = design.four_qubit_design(readout_topology="shared")
        coarse, _, _, _, _ = self._inputs()
        changed = read_json(coarse / "design.json")
        a, b = changed["readout_channels"][:2]
        a["resonator"], b["resonator"] = b["resonator"], a["resonator"]
        self._rebind_changed_design(coarse, changed)
        with self.assertRaisesRegex(ValueError, "Shared readout resonator mapping differs"):
            reduction._case(coarse, "coarse")

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
