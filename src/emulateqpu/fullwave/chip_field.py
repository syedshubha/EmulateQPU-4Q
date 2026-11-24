"""Translate the frozen reference chip layers into zero-thickness EM sheets."""
from __future__ import annotations

import math
from itertools import combinations

from scipy.constants import elementary_charge, h, hbar

from .hardware import HARDWARE_SCHEMA
from .layout import DATABASE_GRID_MM

JUNCTION_SHEET_LENGTH_MM = 0.003
JUNCTION_SHEET_WIDTH_MM = 0.002
CONTACT_TAPER_LENGTH_MM = 0.003
CONTACT_WINDOW_HALF_WIDTH_MM = 0.003


def _replace_junction_tip(island, electrode, bandage, x, y, side, label):
    from shapely import set_precision
    from shapely.geometry import LineString, Polygon, box

    half_length = JUNCTION_SHEET_LENGTH_MM / 2
    half_width = JUNCTION_SHEET_WIDTH_MM / 2
    edge_x = round(x + side * half_length, 6)
    cut_x = round(edge_x + side * CONTACT_TAPER_LENGTH_MM, 6)
    y0, y1 = y - CONTACT_WINDOW_HALF_WIDTH_MM, y + CONTACT_WINDOW_HALF_WIDTH_MM
    section = set_precision(LineString([(cut_x, y0), (cut_x, y1)]), DATABASE_GRID_MM)
    clear = set_precision(box(min(x, cut_x), y0, max(x, cut_x), y1), DATABASE_GRID_MM)
    remaining = island.difference(clear)
    common = remaining.intersection(section)
    if common.geom_type != "LineString" or common.length <= 0:
        raise ValueError(f"Junction {label} has no intact lead for its contact taper")
    _, low, _, high = common.bounds
    if not y0 < low < y < high < y1:
        raise ValueError(f"Junction {label} lead cut does not isolate one device branch")
    taper = set_precision(Polygon([(cut_x, low), (edge_x, y - half_width),
                                  (edge_x, y + half_width), (cut_x, high)]), DATABASE_GRID_MM)
    joined = remaining.union(taper)
    retained = set_precision(box(min(edge_x, cut_x), y0, max(edge_x, cut_x), y1), DATABASE_GRID_MM)
    deposited = electrode.intersection(retained).union(bandage)
    if joined.geom_type != "Polygon" or not joined.is_valid or not joined.covers(deposited):
        raise ValueError(f"Junction {label} contact taper does not contain its cleaned and deposited metal")
    return joined


def _junction_sheets(chip, shapes, metal):
    from shapely import set_precision
    from shapely.geometry import LineString, Point, Polygon, box
    from shapely.ops import unary_union

    modes = chip["mode_order"]
    islands = chip["island_order"]
    devices = chip["junction_devices"]
    loop_rows = {loop["coupler"]: loop for loop in chip["squid_loops"]}
    if len(modes) != 7 or len(islands) != 14 or len(set(islands)) != 14 or len(devices) != 10 or len(loop_rows) != 3:
        raise ValueError("Expected four single junctions and three two-junction SQUIDs")
    island_pairs = {mode: islands[2 * i:2 * i + 2] for i, mode in enumerate(modes)}
    labels_by_mode = {mode: [] for mode in modes}
    patches, junctions = {}, []
    half_length = JUNCTION_SHEET_LENGTH_MM / 2
    half_width = JUNCTION_SHEET_WIDTH_MM / 2
    for index, row in enumerate(devices, start=101):
        label, mode = row["label"], row["mode"]
        island_pair = row["islands"]
        contacts = row["base_contacts"]
        if (mode not in island_pairs or island_pair != island_pairs[mode] or len(contacts) != 2
                or [entry["base_island"] for entry in contacts] != island_pair
                or [entry["electrode"] for entry in contacts] != [row["bottom_electrode"], row["top_electrode"]]):
            raise ValueError(f"Incorrect island or electrode assignment for junction {label}")
        x, y = row["center_mm"]
        patch_name = f"port_jj_{label}"
        patch = set_precision(box(x - half_length, y - half_width, x + half_length, y + half_width),
                              DATABASE_GRID_MM)
        if patch_name in patches or not patch.covers(shapes[row["barrier"]]):
            raise ValueError(f"Unexpected or duplicate tunnel overlap for junction {label}")
        for side, contact in zip((-1, 1), contacts):
            island = contact["base_island"]
            metal[island] = _replace_junction_tip(
                metal[island], shapes[contact["electrode"]], shapes[contact["layer_shape"]], x, y, side, label
            )
        bias = loop_rows[mode]["idle_bias_Phi0"] if mode in loop_rows else 0.0
        energy_GHz = row["junction_energy_GHz"] * math.cos(math.pi * bias)
        capacitance_F = row["capacitance_fF"] * 1e-15
        if not (-0.5 < bias < 0.5 and 0 < energy_GHz < math.inf and 0 < capacitance_F < math.inf):
            raise ValueError(f"Invalid fixed-bias junction parameters for {label}")
        patches[patch_name] = patch
        labels_by_mode[mode].append(label)
        junctions.append({"label": label, "mode": mode, "port": index, "direction": "+X",
                          "islands": list(island_pair), "center_mm": [x, y],
                          "length_mm": JUNCTION_SHEET_LENGTH_MM, "width_mm": JUNCTION_SHEET_WIDTH_MM,
                          "linearized_EJ_GHz": energy_GHz,
                          "inductance_H": (hbar / (2 * elementary_charge))**2 / (h * energy_GHz * 1e9),
                          "capacitance_F": capacitance_F})

    for row in junctions:
        label, (x, y) = row["label"], row["center_mm"]
        patch = patches[f"port_jj_{label}"]
        contacts = {name: patch.intersection(sheet) for name, sheet in metal.items() if patch.intersects(sheet)}
        if set(contacts) != set(row["islands"]):
            raise ValueError(f"Junction sheet {label} must meet only its two islands")
        for side, island in zip((-1, 1), row["islands"]):
            xx = x + side * half_length
            interface = set_precision(LineString([(xx, y - half_width), (xx, y + half_width)]),
                                      DATABASE_GRID_MM)
            if not contacts[island].equals(interface):
                raise ValueError(f"Junction sheet {label} must contact {island} along its whole edge")
    if any(first.intersects(second) for first, second in combinations(patches.values(), 2)):
        raise ValueError("Distinct junction sheets touch in the field geometry")

    for mode, labels in labels_by_mode.items():
        if mode not in loop_rows:
            if labels != [mode]:
                raise ValueError(f"Expected one data junction for {mode}")
            continue
        loop = loop_rows[mode]
        if len(labels) != 2 or labels != loop["junctions"]:
            raise ValueError(f"Unexpected parallel junctions for SQUID {mode}")
        left, right = (metal[name] for name in island_pairs[mode])
        closed = unary_union([left, right, *(patches[f"port_jj_{label}"] for label in labels)])
        center = Point(*loop["center_mm"])
        if (closed.geom_type != "Polygon" or len(closed.interiors) != 1
                or not Polygon(closed.interiors[0]).contains(center)
                or not shapes[loop["aperture"]].contains(center)):
            raise ValueError(f"Field model for SQUID {mode} must retain an aperture between branches")
    return patches, junctions


def _validate_metal_contacts(chip, metal):
    from shapely import set_precision
    from shapely.geometry import LineString

    grounded_flux = {}
    for contact in chip["signal_contacts"]:
        if contact["kind"] == "flux":
            x, y = contact["ground_return_mm"]
            half_width = contact["ground_return_width_mm"] / 2
            grounded_flux[contact["conductor"]] = set_precision(
                LineString([(x - half_width, y), (x + half_width, y)]), DATABASE_GRID_MM
            )
    if any(not metal[name].intersection(metal["chip_ground"]).equals(expected)
           for name, expected in grounded_flux.items()):
        raise ValueError("A flux conductor has an incorrect coplanar ground return")
    for (first_name, first), (second_name, second) in combinations(metal.items(), 2):
        is_flux_return = ("chip_ground" in (first_name, second_name)
                          and (first_name in grounded_flux or second_name in grounded_flux))
        if not is_flux_return and first.intersects(second):
            raise ValueError(f"Unintended PEC contact between {first_name} and {second_name}")


def _bridge_ribbons(chip, shapes, metal):
    from shapely import set_precision
    from shapely.geometry import LineString, box

    output, projections = {}, {}
    ground = metal["chip_ground"]
    for bridge in chip["ground_bridges"]:
        name = bridge["label"]
        strap = shapes[bridge["strap"]]
        feet = [shapes[key] for key in bridge["contacts"]]
        span_axis = bridge["span_axis"]
        height = bridge["clearance_mm"]
        if (name in output or name in metal or len(feet) != 2 or span_axis not in ("x", "y")
                or not 0 < height < chip["enclosure_mm"][5] or not strap.equals(box(*strap.bounds))):
            raise ValueError(f"Invalid ground bridge {name}")
        axis = 0 if span_axis == "x" else 1
        transverse_axis = 1 - axis
        low, high = strap.bounds[axis], strap.bounds[axis + 2]
        near, far = strap.bounds[transverse_axis], strap.bounds[transverse_axis + 2]

        def point(along, across, z):
            return [along, across, z] if axis == 0 else [across, along, z]

        feet.sort(key=lambda shape: shape.centroid.coords[0][axis])
        start, stop = [round(foot.centroid.coords[0][axis], 6) for foot in feet]
        if not low < start < stop < high:
            raise ValueError(f"Ground bridge {name} does not span its contact centers")
        for at, foot in zip((low, high), feet):
            anchor = set_precision(LineString([point(at, near, 0.0)[:2], point(at, far, 0.0)[:2]]),
                                   DATABASE_GRID_MM)
            if not foot.covers(anchor) or not ground.covers(foot):
                raise ValueError(f"Ground bridge {name} misses a grounded contact")
        crossed = {label for label, shape in metal.items()
                   if label != "chip_ground" and strap.intersects(shape)}
        if crossed != {bridge["crosses"]} or any(strap.intersects(shape) for shape in projections.values()):
            raise ValueError(f"Ground bridge {name} crosses unexpected metal or another bridge")

        profile = ((low, 0.0), (start, height), (stop, height), (high, 0.0))
        output[name] = [[point(a, near, za), point(a, far, za), point(b, far, zb), point(b, near, zb)]
                        for (a, za), (b, zb) in zip(profile, profile[1:])]
        projections[name] = strap
    return output


def chip_field_geometry(chip):
    """Return planar polygons, ten lumped junctions, and ground bridge quads.

    All coordinates are in mm. The model binds the final aluminum and cleaned
    contacts into zero-thickness PEC at z=0. A 3 µm contact taper on each side
    contains the bandage and the electrode portion beyond the sheet, and meets
    the 3 µm by 2 µm junction sheet along the full 2 µm width. This homogenizes
    the overlapping tunnel contacts and their local geometric capacitance.
    The oxide capacitance and fixed-bias junction inductance remain explicit
    lumped elements. The two branches per SQUID retain separate sheet ports
    and a loop aperture. Each bridge has a flat elevated section between the
    opening centers and two ramps down to the ends of its mask projection.
    """
    from shapely import from_wkt, set_precision

    if chip["schema"] != HARDWARE_SCHEMA or chip["units"] != "mm":
        raise ValueError("Expected a reference hardware chip record in millimetres")
    shapes = {name: set_precision(from_wkt(wkt), DATABASE_GRID_MM) for name, wkt in chip["sheets_wkt"].items()}
    metal = {name: shapes[name] for name in chip["layout_layers"]["al_base_metal"]["sheets"]}
    patches, junctions = _junction_sheets(chip, shapes, metal)
    _validate_metal_contacts(chip, metal)
    nonplanar = _bridge_ribbons(chip, shapes, metal)
    return {**metal, **patches}, junctions, nonplanar
