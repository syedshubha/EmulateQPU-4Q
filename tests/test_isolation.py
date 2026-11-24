"""Known-answer checks for matched readout and drive reference-plane power."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from emulateqpu.fullwave.design import ASSEMBLY_PORT_ORDER, DATA_QUBITS, PORT_ORDER, SHARED_PORT_ORDER
from emulateqpu.fullwave.isolation import compare_isolation
from emulateqpu.fullwave.reduction import ASSEMBLY_SHARED_PAIR, MODEL_SCHEMA, QUALIFICATION_SCHEMA
from emulateqpu.io import atomic_json, digest, file_hash, read_json


class IsolationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def artifact(self, name, *, shared=False, assembly=False, grid=(5., 6., 7.), adjust=None,
                 model_synthetic=False, report_synthetic=False, design_id=None):
        directory = self.root / name
        directory.mkdir()
        if assembly:
            labels = ASSEMBLY_PORT_ORDER
        else:
            labels = SHARED_PORT_ORDER if shared else PORT_ORDER
        shared_pair = ASSEMBLY_SHARED_PAIR if assembly else ("readout_in", "readout_out")
        channels = []
        for qubit in DATA_QUBITS:
            first, last = shared_pair if shared else (f"readout_{qubit}_in", f"readout_{qubit}_out")
            channels.append({"qubit": qubit, "input_port": first, "output_port": last})
        readout_ports = list(dict.fromkeys(port for channel in channels
                                          for port in (channel["input_port"], channel["output_port"])))
        ports = [{"label": label, "port": i + 1, "Z_ref_ohm": 50., "reference_plane_mm": [i, 0, 0]}
                 for i, label in enumerate(labels)]
        if assembly:
            for port in ports:
                label = port["label"]
                port["active"] = label in readout_ports or not label.startswith("readout_")

        # These matrices are analytical unit fixtures. Their qualification records
        # exercise only the integrity contract, independently of any field solve.
        scattering = np.zeros((len(grid), len(labels), len(labels)), dtype=complex)
        for index, frequency in enumerate(grid):
            for label, amplitude in {"drive_q0": .2, "drive_q1": .1j, "drive_q2": .15,
                                     readout_ports[0]: .02 * frequency, readout_ports[1]: .2j}.items():
                destination = labels.index(label)
                scattering[index, destination, 0] = scattering[index, 0, destination] = amplitude
        if assembly and shared:
            inactive, flux = labels.index("readout_q1_out"), labels.index("flux_c01")
            scattering[:, inactive, 0] = scattering[:, 0, inactive] = .3
            scattering[:, flux, 0] = scattering[:, 0, flux] = .4
        qualification = {"schema": QUALIFICATION_SCHEMA, "passed": True,
                         "design_sha256": digest(design_id or name), "synthetic": report_synthetic}
        model = {"schema": MODEL_SCHEMA, "qualification_sha256": digest(qualification),
                 "design_sha256": qualification["design_sha256"], "synthetic": model_synthetic,
                 "ports": ports, "readout_channels": channels, "frequency_grid_GHz": list(grid),
                 "scattering_real": scattering.real.tolist(), "scattering_imag": scattering.imag.tolist()}
        if shared or assembly:
            model["readout_topology"] = "shared" if shared else "dedicated"
        if adjust:
            adjust(model)
        atomic_json(directory / "qualification.json", qualification)
        atomic_json(directory / "model.json", model)
        atomic_json(directory / "electrostatic-cell.json", {})
        atomic_json(directory / "artifact.json", {
            "schema": "emulateqpu.fullwave.artifact.v1", "passed": True,
            "files_sha256": {path.name: file_hash(path) for path in directory.iterdir()},
        })
        return directory

    def test_matched_samples_and_per_drive_power(self):
        dedicated = self.artifact("dedicated", grid=(5., 6., 7., 8.))
        shared = self.artifact("shared", shared=True, grid=(4., 5., 6., 7.))
        report = compare_isolation([dedicated, shared], band_GHz=(6., 7.))
        self.assertEqual(report["frequency_grid_GHz"], [6., 7.])
        self.assertEqual(report["requested_band_GHz"], [6., 7.])
        self.assertEqual(report["drive_ports"], ["drive_q0", "drive_q1", "drive_q2", "drive_q3"])
        for row, topology, expected_ports in zip(report["models"], ("dedicated", "shared"),
                                                  (list(PORT_ORDER[4:]), list(SHARED_PORT_ORDER[4:])), strict=True):
            self.assertEqual(row["readout_topology"], topology)
            self.assertEqual(row["reference_plane_kind"], "die")
            self.assertEqual(row["readout_ports"], expected_ports)
            self.assertEqual(row["artifact_manifest_sha256"], file_hash(Path(row["artifact_path"]) / "artifact.json"))
            drive = row["drives"]["drive_q0"]
            np.testing.assert_allclose(drive["readout_power_fraction"], [.12**2 + .2**2, .14**2 + .2**2])
            np.testing.assert_allclose(drive["reflected_power_fraction"], [.04, .04])
            np.testing.assert_allclose(drive["other_drive_power_fraction"], [.1**2 + .15**2] * 2)
            np.testing.assert_allclose(drive["spectator_drive_power_fraction"]["drive_q1"], [.01, .01])
            self.assertEqual(drive["max_readout_frequency_GHz"], 7.)
            self.assertAlmostEqual(drive["max_other_drive_power_fraction"], .0325)
        json.dumps(report, allow_nan=False)

    def test_shared_carrier_excludes_unbonded_readout_and_flux_ports(self):
        dedicated = self.artifact("dedicated", assembly=True)
        shared = self.artifact("shared", shared=True, assembly=True)
        report = compare_isolation([dedicated, shared])
        row = report["models"][1]
        self.assertEqual(row["reference_plane_kind"], "carrier")
        self.assertEqual(row["readout_ports"], list(ASSEMBLY_SHARED_PAIR))
        np.testing.assert_allclose(row["drives"]["drive_q0"]["readout_power_fraction"],
                                   [.10**2 + .20**2, .12**2 + .20**2, .14**2 + .20**2])

    def test_missing_duplicate_and_unmatched_input(self):
        first = self.artifact("first", design_id="one design")
        copied = self.artifact("copy", design_id="one design")
        shifted = self.artifact("shifted", grid=(5.5, 6.5, 7.5))
        with self.assertRaisesRegex(ValueError, "distinct qualified artifact directories"):
            compare_isolation([first, first])
        with self.assertRaisesRegex(ValueError, "distinct qualified artifact directories"):
            compare_isolation(first)
        with self.assertRaisesRegex(ValueError, "distinct full-wave designs"):
            compare_isolation([first, copied])
        with self.assertRaisesRegex(ValueError, "No exactly matching solved frequencies"):
            compare_isolation([first, shifted])
        with self.assertRaisesRegex(ValueError, "increasing bounds"):
            compare_isolation([first, shifted], band_GHz=(7., 6.))

    def test_legacy_malformed_or_unmatched_ports_are_rejected(self):
        first = self.artifact("first")
        legacy = self.artifact("legacy", shared=True, adjust=lambda model: model.pop("readout_channels"))
        broken = self.artifact("broken", adjust=lambda model: model["scattering_real"].pop())
        mounted = self.artifact("mounted", assembly=True)
        moved = self.artifact("moved", adjust=lambda model: model["ports"][0]["reference_plane_mm"].__setitem__(0, 1))
        for artifact, message in ((legacy, "four physically mapped readout channels"), (broken, "Scattering table shape"),
                                  (mounted, "matched 50-ohm reference planes"), (moved, "matched 50-ohm reference planes")):
            with self.subTest(artifact=artifact), self.assertRaisesRegex(ValueError, message):
                compare_isolation([first, artifact])

    def test_hash_integrity_and_synthetic_evidence(self):
        first = self.artifact("first")
        model_fixture = self.artifact("model-fixture", model_synthetic=True)
        report_fixture = self.artifact("report-fixture", report_synthetic=True)
        for artifact in (model_fixture, report_fixture):
            with self.assertRaisesRegex(ValueError, "non-synthetic"):
                compare_isolation([first, artifact])
        modified = self.artifact("modified")
        (modified / "model.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            compare_isolation([first, modified])

    def test_nonfinite_values_are_rejected_before_power_conversion(self):
        first = self.artifact("first")
        second = self.artifact("second")
        model = read_json(first / "model.json")
        model["scattering_imag"][0][0][0] = float("nan")
        with patch("emulateqpu.fullwave.isolation.load_model", return_value=model):
            with self.assertRaisesRegex(ValueError, "finite numeric RF values"):
                compare_isolation([first, second])
