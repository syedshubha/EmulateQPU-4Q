"""Export planar chip geometry to a layered GDSII design review package."""
from __future__ import annotations

from datetime import datetime
from math import isfinite
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from ..io import atomic_json, digest, file_hash
from .design import DESIGN_SCHEMA, LEGACY_DESIGN_SCHEMA, four_qubit_design

CELL_NAME = "EMULATEQPU_4Q"
USER_UNIT_M = 1e-6
DATABASE_UNIT_M = 1e-9
DATABASE_GRID_MM = DATABASE_UNIT_M * 1e3
DATABASE_GRID_UM = DATABASE_UNIT_M / USER_UNIT_M
MAX_BOUNDARY_VERTICES = 199
OUTLINE_SHEET = "layout_chip_outline"


def _default_layers(design, footprint):
    from shapely.geometry import box

    if design["schema"] not in (DESIGN_SCHEMA, LEGACY_DESIGN_SCHEMA):
        raise ValueError("An explicit layout layer map is required for this design schema")
    rf_labels = {f"port_{p['label']}": p["label"] for p in design["ports"]}
    junction_labels = {f"port_jj_{j['label']}": f"JJ {j['label']}" for j in design["junctions"]}
    layers = {
        "top_metal": {"layer": 1, "datatype": 0, "fabrication": True,
                      "sheets": design["conductors"]},
        "chip_outline": {"layer": 100, "datatype": 0, "fabrication": False,
                         "sheets": [OUTLINE_SHEET]},
        "rf_port_marker": {"layer": 101, "datatype": 0, "fabrication": False,
                           "sheets": list(rf_labels), "labels": rf_labels},
        "junction_marker": {"layer": 102, "datatype": 0, "fabrication": False,
                            "sheets": list(junction_labels), "labels": junction_labels},
    }
    derived_sheets = {OUTLINE_SHEET: footprint.wkt}
    bridge_labels = {}
    for bridge in design.get("ground_bridges", []):
        label = bridge["label"]
        name = f"layout_{label}"
        center_x, _ = bridge["center_mm"]
        half_width = bridge["width_mm"] / 2
        y_values, _ = zip(*bridge["profile_yz_mm"])
        y_min, y_max = min(y_values), max(y_values)
        if (name in derived_sheets or half_width <= 0 or y_min >= y_max
                or not all(isfinite(value) for value in (center_x, half_width, *y_values))):
            raise ValueError(f"Invalid projected ground bridge {label}")
        derived_sheets[name] = box(center_x - half_width, y_min, center_x + half_width, y_max).wkt
        bridge_labels[name] = label
    if bridge_labels:
        layers["ground_bridge_marker"] = {"layer": 103, "datatype": 0, "fabrication": False,
                                          "sheets": list(bridge_labels), "labels": bridge_labels}
    return layers, derived_sheets


def _validate_layers(layers, sheets):
    if not isinstance(layers, dict) or not layers:
        raise ValueError("Expected a nonempty layout layer map")
    assigned, used_layers = set(), set()
    for purpose, entry in layers.items():
        if (not isinstance(purpose, str) or not purpose or not isinstance(entry, dict)
                or set(entry) - {"layer", "datatype", "sheets", "fabrication", "labels"}):
            raise ValueError(f"Invalid layout layer purpose {purpose!r}")
        spec = (entry.get("layer"), entry.get("datatype"))
        if any(type(index) is not int or not 0 <= index <= 255 for index in spec) or spec in used_layers:
            raise ValueError(f"Invalid or duplicate GDS layer/datatype for {purpose}")
        used_layers.add(spec)
        names = entry.get("sheets")
        if (not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names)
                or len(names) != len(set(names)) or set(names) & assigned or set(names) - sheets.keys()):
            raise ValueError(f"Missing or multiply assigned sheets in layout layer {purpose}")
        assigned.update(names)
        labels = entry.get("labels", {})
        if (type(entry.get("fabrication")) is not bool or not isinstance(labels, dict)
                or (entry["fabrication"] and labels) or set(labels) - set(names)
                or any(not isinstance(text, str) or not text.isascii() or not text.isprintable()
                       or not 0 < len(text) <= 512 for text in labels.values())):
            raise ValueError(f"Invalid label or fabrication role for layout layer {purpose}")
    if assigned != sheets.keys():
        raise ValueError(f"Unassigned layout sheets: {sorted(sheets.keys() - assigned)}")


def _polygons(geometry):
    return geometry.geoms if geometry.geom_type == "MultiPolygon" else (geometry,)


def _sheet_geometry(text, name, footprint):
    from shapely import from_wkt, get_coordinate_dimension, set_precision

    shape = from_wkt(text, on_invalid="ignore") if isinstance(text, str) else None
    if (shape is None or shape.geom_type not in ("Polygon", "MultiPolygon") or shape.is_empty
            or not shape.is_valid or get_coordinate_dimension(shape) != 2):
        raise ValueError(f"Layout sheet {name} must be a valid nonempty planar polygon")
    snapped = set_precision(shape, DATABASE_GRID_MM)
    rings = sorted(len(part.interiors) for part in _polygons(shape))
    if (snapped.is_empty or sorted(len(part.interiors) for part in _polygons(snapped)) != rings
            or not footprint.covers(snapped)):
        raise ValueError(f"Layout sheet {name} changed topology on the 1 nm grid or leaves the chip outline")
    return snapped


def _gds_polygons(gdstk, geometry, layer, datatype):
    def ring_polygon(ring):
        return gdstk.Polygon([(x * 1000, y * 1000) for x, y in ring.coords[:-1]],
                             layer=layer, datatype=datatype)

    for polygon in _polygons(geometry):
        shell = ring_polygon(polygon.exterior)
        if polygon.interiors:
            holes = [ring_polygon(ring) for ring in polygon.interiors]
            yield from gdstk.boolean(shell, holes, "not", precision=DATABASE_GRID_UM,
                                     layer=layer, datatype=datatype)
        else:
            yield shell


def export_layout(output, *, design=None):
    """Write a GDSII stream, its source design, and a hash-bound layer inventory."""
    import gdstk
    from shapely import set_precision
    from shapely.geometry import box

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if design is None:
        design, _ = four_qubit_design()
    if (not isinstance(design, dict) or design.get("units") != "mm"
            or not isinstance(design.get("schema"), str) or not isinstance(design.get("design_id"), str)):
        raise ValueError("Expected a chip design record with millimetre coordinates")
    cell_name = design.get("gds_top_cell", CELL_NAME)
    if not isinstance(cell_name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", cell_name):
        raise ValueError("Invalid GDSII top cell name")
    bounds = design.get("enclosure_mm")
    if (not isinstance(bounds, list) or len(bounds) != 6
            or any(type(value) not in (int, float) or not isfinite(value) for value in bounds)
            or any(lo >= hi for lo, hi in zip(bounds[:3], bounds[3:]))
            or max(abs(bounds[i]) for i in (0, 1, 3, 4)) / DATABASE_GRID_MM >= 2**31 - 1):
        raise ValueError("Invalid chip enclosure bounds for a 1 nm GDSII grid")
    footprint = set_precision(box(bounds[0], bounds[1], bounds[3], bounds[4]), DATABASE_GRID_MM)
    sheets = design.get("sheets_wkt")
    if footprint.is_empty or not isinstance(sheets, dict) or not sheets:
        raise ValueError("Expected a chip footprint and planar layout sheets")
    layers = design.get("layout_layers")
    derived_sheets = {}
    if layers is None:
        layers, derived_sheets = _default_layers(design, footprint)
        if conflict := sheets.keys() & derived_sheets.keys():
            raise ValueError(f"Reserved layout sheet names: {sorted(conflict)}")
        sheets = {**sheets, **derived_sheets}
    _validate_layers(layers, sheets)

    library = gdstk.Library(name=cell_name, unit=USER_UNIT_M, precision=DATABASE_UNIT_M)
    cell = library.new_cell(cell_name)
    layer_order = sorted(layers.values(), key=lambda entry: (entry["layer"], entry["datatype"]))
    for entry in layer_order:
        layer, datatype = entry["layer"], entry["datatype"]
        labels = entry.get("labels", {})
        for name in sorted(entry["sheets"]):
            geometry = _sheet_geometry(sheets[name], name, footprint)
            cell.add(*_gds_polygons(gdstk, geometry, layer, datatype))
            if name in labels:
                point = geometry.representative_point()
                cell.add(gdstk.Label(labels[name], (point.x * 1000, point.y * 1000),
                                     layer=layer, texttype=datatype))

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".layout-", dir=output.parent) as directory:
        staged = Path(directory)
        library.write_gds(staged / "layout.gds", max_points=MAX_BOUNDARY_VERTICES,
                          timestamp=datetime(2000, 1, 1))
        atomic_json(staged / "design.json", design)
        manifest = {
            "schema": "emulateqpu.fullwave.layout.v1", "status": "LAYOUT_EXPORTED",
            "format": "GDSII", "top_cell": cell_name, "database_unit_m": DATABASE_UNIT_M,
            "user_unit_m": USER_UNIT_M, "gdstk_version": gdstk.__version__,
            "design_id": design["design_id"], "design_sha256": digest(design),
            "files_sha256": {name: file_hash(staged / name) for name in ("layout.gds", "design.json")},
            "layers": layers, "derived_sheets_wkt": derived_sheets, "foundry_drc_qualified": False,
            "scope": "Positive layout polygons and review markers with the source geometry and layer mapping",
        }
        atomic_json(staged / "layout.json", manifest)
        output.mkdir()
        for name in ("layout.gds", "design.json", "layout.json"):
            (staged / name).replace(output / name)
    return manifest
