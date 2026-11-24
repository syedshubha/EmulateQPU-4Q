"""Round-trip the generated GDSII with an independent geometry comparison."""
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from emulateqpu.fullwave.__main__ import main as fullwave_main
from emulateqpu.fullwave import design as field_design
from emulateqpu.fullwave.design import DEFAULT_READOUT_FREQUENCIES_GHZ, MODE_ORDER, PORT_ORDER, four_qubit_design
from emulateqpu.fullwave.layout import CELL_NAME, DATABASE_UNIT_M, USER_UNIT_M, export_layout
from emulateqpu.io import digest, file_hash, read_json


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("gdstk", "shapely")),
                     "Physical layout dependencies required")
class GdsLayoutTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _read(self, output):
        import gdstk

        library = gdstk.read_gds(output / "layout.gds")
        self.assertAlmostEqual(library.unit, USER_UNIT_M, places=15)
        self.assertAlmostEqual(library.precision, DATABASE_UNIT_M, places=15)
        cells = library.top_level()
        self.assertEqual([cell.name for cell in cells], [CELL_NAME])
        self.assertTrue(all(len(poly.points) <= 199 for poly in cells[0].polygons))
        return cells[0]

    def _layer_shape(self, cell, layer, datatype=0):
        from shapely import set_precision
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        return set_precision(unary_union([
            Polygon(poly.points * .001).buffer(0) for poly in cell.polygons
            if (poly.layer, poly.datatype) == (layer, datatype)
        ]), 1e-6)

    def test_four_line_gds_matches_source_and_keeps_markers_off_metal(self):
        from shapely import unary_union

        design, sheets = four_qubit_design()
        output = self.root / "layout"
        manifest = export_layout(output)
        self.assertEqual(read_json(output / "layout.json"), manifest)
        self.assertEqual(read_json(output / "design.json"), design)
        self.assertEqual(manifest["design_sha256"], digest(design))
        self.assertFalse(manifest["foundry_drc_qualified"])
        for name, sha in manifest["files_sha256"].items():
            self.assertEqual(file_hash(output / name), sha)
        self.assertEqual({purpose for purpose, row in manifest["layers"].items() if row["fabrication"]},
                         {"top_metal"})
        cell = self._read(output)
        metal = self._layer_shape(cell, 1)
        expected_metal = unary_union([sheets[name] for name in design["conductors"]])
        self.assertAlmostEqual(metal.symmetric_difference(expected_metal).area, 0., places=10)
        self.assertEqual(len(metal.geoms), len(expected_metal.geoms))
        self.assertGreater(len(sheets["chip_ground"].geoms), 1)
        self.assertEqual({label.text for label in cell.labels if label.layer == 101}, set(PORT_ORDER))
        self.assertEqual({label.text for label in cell.labels if label.layer == 102},
                         {f"JJ {mode}" for mode in MODE_ORDER})
        for marker_layer in (101, 102):
            self.assertAlmostEqual(self._layer_shape(cell, marker_layer).intersection(metal).area, 0.)
        second = export_layout(self.root / "second", design=read_json(output / "design.json"))
        self.assertEqual(second, manifest)
        with self.assertRaises(FileExistsError):
            export_layout(output, design=design)
        self.assertEqual(file_hash(output / "layout.gds"), manifest["files_sha256"]["layout.gds"])

    def test_custom_mask_layers_preserve_ground_holes_and_separate_straps(self):
        from shapely.geometry import MultiPolygon, box

        ground = MultiPolygon([box(0., 0., 3., 3.).difference(box(1., 1., 2., 2.)),
                               box(4., 0., 5., 1.)])
        strap = box(2.8, .3, 4.2, .4)
        landing = box(2.8, .25, 3., .45)
        marker = box(1.2, 1.2, 1.8, 1.8)
        design = {
            "schema": "synthetic.hardware", "units": "mm", "design_id": "test-mask-layers",
            "enclosure_mm": [0., 0., -0.35, 5., 4., .7],
            "sheets_wkt": {name: shape.wkt for name, shape in {
                "ground": ground, "strap": strap, "landing": landing, "marker": marker}.items()},
            "layout_layers": {
                "top_metal": {"layer": 1, "datatype": 0, "fabrication": True, "sheets": ["ground"]},
                "bridge_strap": {"layer": 3, "datatype": 0, "fabrication": True, "sheets": ["strap"]},
                "bridge_landing": {"layer": 3, "datatype": 1, "fabrication": True, "sheets": ["landing"]},
                "junction": {"layer": 102, "datatype": 1, "fabrication": False, "sheets": ["marker"],
                             "labels": {"marker": "JJ q0"}},
            },
        }
        output = self.root / "explicit"
        export_layout(output, design=design)
        cell = self._read(output)
        read_ground = self._layer_shape(cell, 1)
        self.assertEqual(len(read_ground.geoms), 2)
        self.assertEqual(sum(len(part.interiors) for part in read_ground.geoms), 1)
        for layer, datatype, expected in ((1, 0, ground), (3, 0, strap), (3, 1, landing), (102, 1, marker)):
            self.assertAlmostEqual(self._layer_shape(cell, layer, datatype).symmetric_difference(expected).area,
                                   0., places=10)
        self.assertEqual([(label.layer, label.texttype, label.text) for label in cell.labels], [(102, 1, "JJ q0")])
        unassigned = deepcopy(design)
        del unassigned["layout_layers"]["junction"]
        with self.assertRaisesRegex(ValueError, "Unassigned layout sheets"):
            export_layout(self.root / "unassigned", design=unassigned)
        duplicate = deepcopy(design)
        duplicate["layout_layers"]["bridge_strap"]["sheets"].append("ground")
        with self.assertRaisesRegex(ValueError, "multiply assigned"):
            export_layout(self.root / "duplicate", design=duplicate)

    def test_legacy_design_retains_six_rf_ports(self):
        design, _ = four_qubit_design(legacy=True)
        output = self.root / "legacy"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(fullwave_main(["layout", str(output), "--legacy"]), 0)
        manifest = read_json(output / "layout.json")
        self.assertEqual(manifest["design_id"], design["design_id"])
        self.assertNotIn("chip_ground", manifest["layers"]["top_metal"]["sheets"])
        self.assertEqual(len([label for label in self._read(output).labels if label.layer == 101]), 6)

    def test_shared_layout_cli_accepts_existing_design_or_topology(self):
        from shapely import from_wkt, unary_union

        output = self.root / "shared"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(fullwave_main(["layout", str(output), "--readout-topology", "shared"]), 0)
        shared, _ = four_qubit_design(readout_topology="shared")
        source = output / "design.json"
        self.assertEqual(read_json(source), shared)
        cell = self._read(output)
        self.assertEqual(len(shared["readout_channels"]), 4)
        self.assertEqual({label.text for label in cell.labels if label.layer == 101},
                         {port["label"] for port in shared["ports"]})
        bridge_layer = read_json(output / "layout.json")["layers"]["ground_bridge_marker"]
        self.assertFalse(bridge_layer["fabrication"])
        self.assertEqual({label.text for label in cell.labels if label.layer == 103},
                         {bridge["label"] for bridge in shared["ground_bridges"]})
        derived = read_json(output / "layout.json")["derived_sheets_wkt"]
        expected_bridges = unary_union([from_wkt(derived[name]) for name in bridge_layer["sheets"]])
        bridge_shape = self._layer_shape(cell, 103)
        self.assertGreater(bridge_shape.area, 0.)
        self.assertAlmostEqual(bridge_shape.symmetric_difference(expected_bridges).area, 0., places=10)

        reexport = self.root / "reexport"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(fullwave_main(["layout", str(reexport), "--design", str(source)]), 0)
        self.assertEqual(file_hash(output / "layout.gds"), file_hash(reexport / "layout.gds"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            fullwave_main(["layout", str(self.root / "conflict"), "--design", str(source),
                           "--readout-topology", "shared"])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / "conflict").exists())

    def test_planar_controls_bind_the_same_moved_geometry_to_gds_and_field_inputs(self):
        def mesh(record, sheets, path, refinement):
            path.write_text(refinement, encoding="utf-8")
            bridges = [item["label"] for item in record.get("ground_bridges", ())]
            groups = {name: i for i, name in enumerate(["enclosure", *sheets, *bridges], 1)}
            return groups, {"substrate": 514, "air": 501}, 1000

        frequencies = [6.40, 6.62, 6.84, 7.08]
        for topology in ("dedicated", "shared"):
            with self.subTest(topology=topology):
                output, prepared = self.root / f"{topology}-layout", self.root / f"{topology}-coarse"
                args = ["--readout-topology", topology, "--drive-gap-um", "40",
                        "--readout-frequencies-ghz", *map(str, frequencies)]
                with redirect_stdout(io.StringIO()), patch.object(field_design, "_mesh_design", side_effect=mesh):
                    self.assertEqual(fullwave_main(["layout", str(output), *args]), 0)
                    self.assertEqual(fullwave_main(["prepare", str(prepared), *args]), 0)
                variant, sheets = four_qubit_design(readout_topology=topology, drive_gap_um=40,
                                                     readout_frequencies_GHz=frequencies)
                baseline, baseline_sheets = four_qubit_design(readout_topology=topology)
                explicit_defaults, _ = four_qubit_design(readout_topology=topology, drive_gap_um=25,
                                                          readout_frequencies_GHz=DEFAULT_READOUT_FREQUENCIES_GHZ)
                self.assertEqual(baseline, explicit_defaults)
                self.assertEqual(variant["ports"], baseline["ports"])
                self.assertEqual(read_json(output / "design.json"), variant)
                self.assertEqual(read_json(prepared / "field-inputs.json")["design_sha256"], digest(variant))
                self.assertEqual(read_json(prepared / "design.json"), variant)
                for i, row in enumerate(variant["readout_channels"]):
                    self.assertAlmostEqual(sheets[f"drive_q{i}"].distance(sheets[f"q{2*i}_a"]) * 1000, 40)
                    self.assertEqual(row["nominal_frequency_GHz"], frequencies[i])
                    self.assertFalse(sheets[row["resonator"]].equals(baseline_sheets[row["resonator"]]))

    def test_bad_planar_controls_fail_before_writing(self):
        cases = [
            ("layout", ["--legacy", "--drive-gap-um", "40"]),
            ("layout", ["--hardware", "--drive-gap-um", "40"]),
            ("prepare", ["--drive-gap-um", "nan"]),
            ("prepare", ["--readout-frequencies-ghz", "6", "6.6", "6.8", "10"]),
        ]
        for name, args in cases:
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                fullwave_main([name, str(self.root / "bad"), *args])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((self.root / "bad").exists())
