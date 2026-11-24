"""Check the paired multilayer designs and their GDSII export."""
import contextlib
import importlib.util
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from emulateqpu.fullwave.__main__ import main
from emulateqpu.fullwave.design import four_qubit_design
from emulateqpu.fullwave.hardware import reference_hardware_design
from emulateqpu.io import digest, read_json


@unittest.skipUnless(importlib.util.find_spec("shapely"), "Physical layout dependencies required")
class ReferenceHardwareTests(unittest.TestCase):
    def test_matched_junctions_and_filtered_readout_paths(self):
        from shapely import from_wkt

        dedicated, shared = (reference_hardware_design(readout_topology=topology)
                             for topology in ("dedicated", "shared"))
        for design, count in ((dedicated, 15), (shared, 9)):
            source, _ = four_qubit_design(readout_topology=design["readout_topology"])
            self.assertEqual(design["field_prototype_design_sha256"], digest(source))
            self.assertEqual(len(design["signal_contacts"]), count)
            self.assertEqual({row["coupler"] for row in design["signal_contacts"] if row["kind"] == "flux"},
                             {"c01", "c12", "c23"})
            self.assertEqual(len(design["junction_devices"]), 10)
            self.assertEqual(len(design["squid_loops"]), 3)
            self.assertEqual(len(design["layout_layers"]["al_junction_bandage"]["sheets"]), 20)
            self.assertEqual(len(design["purcell_filters"]), 4)
            shapes = {name: from_wkt(wkt) for name, wkt in design["sheets_wkt"].items()}
            for row in design["purcell_filters"]:
                resonator, purcell, feedline = (shapes[row[key]] for key in ("resonator", "filter", "feedline"))
                self.assertAlmostEqual(resonator.distance(purcell), 0.010)
                self.assertAlmostEqual(purcell.distance(feedline), 0.010)
                self.assertGreater(resonator.distance(feedline), 0.075)
            for row in design["junction_devices"]:
                for contact in row["base_contacts"]:
                    bandage = shapes[contact["layer_shape"]]
                    self.assertGreater(bandage.intersection(shapes[contact["base_island"]]).area, 0.0)
                    self.assertGreater(bandage.intersection(shapes[contact["electrode"]]).area, 0.0)
                    self.assertFalse(bandage.intersects(shapes[row["barrier"]]))
        self.assertEqual(dedicated["junction_devices"], shared["junction_devices"])
        self.assertEqual(dedicated["squid_loops"], shared["squid_loops"])
        for row in dedicated["purcell_filters"]:
            for name in (row["filter"], row["resonator"]):
                self.assertEqual(dedicated["sheets_wkt"][name], shared["sheets_wkt"][name])

    @unittest.skipUnless(importlib.util.find_spec("gdstk"), "GDSII dependency required")
    def test_hardware_layout_cli_writes_signal_labels_and_metal_layers(self):
        import gdstk

        with TemporaryDirectory() as directory:
            for topology in ("dedicated", "shared"):
                with self.subTest(topology=topology):
                    output = Path(directory) / topology
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main(["layout", str(output), "--hardware",
                                               "--readout-topology", topology]), 0)
                    design = read_json(output / "design.json")
                    manifest = read_json(output / "layout.json")
                    self.assertEqual(manifest["design_sha256"], digest(design))
                    self.assertFalse(manifest["foundry_drc_qualified"])
                    cell, = gdstk.read_gds(output / "layout.gds").top_level()
                    self.assertEqual(cell.name, f"QPU4_{topology.upper()}_DIE_R1")
                    self.assertEqual({label.text for label in cell.labels if label.layer == 101},
                                     {row["label"] for row in design["signal_contacts"]})
                    self.assertEqual(len([poly for poly in cell.polygons if poly.layer == 6]), 20)
                    self.assertEqual(len([poly for poly in cell.polygons if poly.layer == 5]),
                                     len(design["ground_bridges"]))

    @unittest.skipUnless(importlib.util.find_spec("gdstk"), "GDSII dependency required")
    def test_carrier_pairs_both_dies_to_one_ground_signal_ground_fanout(self):
        import gdstk

        with TemporaryDirectory() as directory:
            output = Path(directory) / "carrier"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["carrier", str(output)]), 0)
            design = read_json(output / "design.json")
            shared = design["wire_bond_assemblies"]["shared"]
            dedicated = design["wire_bond_assemblies"]["dedicated"]
            targets = {row["die_site"]: row["carrier_site"] for row in shared["bonded"]}
            self.assertEqual((targets["readout_in"], targets["readout_out"]),
                             ("readout_q0_in", "readout_q3_out"))
            self.assertEqual((len(dedicated["bonded"]), len(shared["bonded"])), (15, 9))
            self.assertEqual(set(dedicated["unbonded_carrier_sites"]), set())
            self.assertEqual(set(shared["unbonded_carrier_sites"]),
                             {row["label"] for row in design["carrier_sites"]} - set(targets.values()))
            self.assertEqual(shared["unbonded_outer_loads_ohm"],
                             dict.fromkeys(shared["unbonded_carrier_sites"], 50.0))
            for assembly in (dedicated, shared):
                for site in assembly["bonded"]:
                    self.assertEqual(len(site["ground_wires_mm"]), 2)
                    for wire in (site["signal_wire_mm"], *site["ground_wires_mm"]):
                        self.assertEqual([point[2] for point in wire], [0.0, 0.1, 0.0])
                        self.assertEqual(len({point[0] for point in wire}), 1)
            cell, = gdstk.read_gds(output / "layout.gds").top_level()
            self.assertEqual(cell.name, "QPU4_COMMON_CARRIER_R1")
            self.assertEqual({label.text for label in cell.labels if label.layer == 101},
                             {row["label"] for row in design["carrier_sites"]})
