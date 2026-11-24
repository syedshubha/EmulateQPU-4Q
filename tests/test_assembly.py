"""Check the assembled field model wiring and frozen inputs."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from emulateqpu.fullwave.__main__ import main
from emulateqpu.fullwave.assembly import ASSEMBLY_SOURCES, assembly_geometry
from emulateqpu.fullwave.assembly_mesh import MATERIAL_IDS
from emulateqpu.fullwave.carrier_layout import reference_carrier_design
from emulateqpu.fullwave.design import ASSEMBLY_PORT_ORDER, _configs
from emulateqpu.fullwave.hardware import reference_hardware_design
from emulateqpu.fullwave.reduction import _case
from emulateqpu.io import atomic_json, digest, file_hash, read_json


@unittest.skipUnless(importlib.util.find_spec("shapely"), "Physical layout dependencies required")
class AssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dedicated = reference_hardware_design()
        cls.shared = reference_hardware_design(readout_topology="shared")
        cls.carrier = reference_carrier_design(cls.dedicated, cls.shared)

    def test_common_ports_and_balanced_elements_on_both_comparisons(self):
        for chip in (self.dedicated, self.shared):
            with self.subTest(topology=chip["readout_topology"]):
                design, planar, lifted, _ = assembly_geometry(chip, self.carrier)
                groups = {name: i for i, name in enumerate(("enclosure", *planar, *lifted), 1)}
                configs = _configs(design, groups, MATERIAL_IDS, 40, [4., 9.], [5., 8.])
                self.assertEqual(tuple(row["label"] for row in design["ports"]), ASSEMBLY_PORT_ORDER)
                self.assertEqual({row["label"] for row in design["ports"] if not row["active"]},
                                 set(self.carrier["wire_bond_assemblies"][chip["readout_topology"]]
                                     ["unbonded_carrier_sites"]))
                self.assertEqual(len(lifted), len(chip["signal_contacts"]) * 3 + len(chip["ground_bridges"]))
                boundary = configs["driven-train"]["Boundaries"]
                self.assertEqual(len(boundary["LumpedPort"]), 25)
                for n, row in enumerate(boundary["LumpedPort"][:15], 1):
                    self.assertEqual((row["Index"], row["Excitation"], row["R"]), (n, n, 50.0))
                    self.assertEqual([p["Direction"] for p in row["Elements"]], ["+X", "-X"])
                for row in boundary["LumpedPort"][15:]:
                    self.assertGreater(row["C"], 0)
                    self.assertGreater(row["L"], 0)
                    self.assertNotIn("Excitation", row)
                self.assertEqual(len(configs["electrostatic"]["Boundaries"]["Terminal"]), 14)
                self.assertEqual(len(configs["electrostatic"]["Domains"]["Materials"]), 4)

    def test_prepare_assembly_binds_both_frozen_layouts(self):
        def mesh(path, *, planar, nonplanar, **kwargs):
            self.assertEqual(kwargs["refinement"], "coarse")
            path.write_bytes(b"fixture mesh")
            names = ("enclosure", *planar, *nonplanar)
            return {name: i for i, name in enumerate(names, 1)}, MATERIAL_IDS, 25

        with TemporaryDirectory() as directory:
            root = Path(directory)
            chip_path, carrier_path = root / "chip.json", root / "carrier.json"
            atomic_json(chip_path, self.shared)
            atomic_json(carrier_path, self.carrier)
            out = root / "case"
            with patch("emulateqpu.fullwave.assembly_mesh.mesh_assembly", side_effect=mesh):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(["prepare-assembly", str(out), "--readout-topology", "shared",
                                           "--chip-layout", str(chip_path), "--carrier-layout", str(carrier_path),
                                           "--flux-transfer"]), 0)
            design = read_json(out / "design.json")
            manifest = read_json(out / "field-inputs.json")
            self.assertEqual(manifest["design_sha256"], digest(design))
            self.assertEqual(design["chip_design_sha256"], digest(self.shared))
            self.assertEqual(design["carrier_design_sha256"], digest(self.carrier))
            self.assertTrue(set(ASSEMBLY_SOURCES).issubset(manifest["files_sha256"]))
            self.assertEqual({(row["input_port"], row["output_port"]) for row in design["readout_channels"]},
                             {("readout_q0_in", "readout_q3_out")})
            self.assertEqual(_case(out, "coarse")[2], design)
            self.assertEqual(manifest["expected_outputs"]["magnetostatic"], ["surface-F.csv", "terminal-I.csv"])
            config_path = out / "magnetostatic.json"
            config = read_json(config_path)
            boundaries = config["Boundaries"]
            currents = boundaries["SurfaceCurrent"]
            pickups = boundaries["Postprocessing"]["SurfaceFlux"]
            self.assertEqual([row["Index"] for row in currents], [1, 2, 3])
            self.assertEqual([row["Index"] for row in pickups], [1, 2, 3])
            for current, pickup in zip(currents, pickups):
                self.assertEqual([part["Direction"] for part in current["Elements"]], ["+X", "-X"])
                self.assertEqual((pickup["Type"], pickup["TwoSided"], pickup["Center"][-1]), ("Magnetic", False, -.1))
                self.assertTrue(set(pickup["Attributes"]).isdisjoint(boundaries["PEC"]["Attributes"]))
                self.assertTrue(all(set(part["Attributes"]).isdisjoint(boundaries["PEC"]["Attributes"])
                                    for part in current["Elements"]))
            self.assertNotIn("FluxLoop", boundaries)
            pickups[0]["Center"][-1] = .1
            atomic_json(config_path, config)
            manifest["files_sha256"]["magnetostatic.json"] = file_hash(config_path)
            atomic_json(out / "field-inputs.json", manifest)
            with self.assertRaisesRegex(ValueError, "magnetostatic inputs differ"):
                _case(out, "coarse")
        changed = deepcopy(self.shared)
        changed["scope"] = "different frozen layout"
        with self.assertRaisesRegex(ValueError, "differs from the carrier"):
            assembly_geometry(changed, self.carrier)
