"""Fabrication-layout RF bond pads, flux bias routes, and coplanar ground."""
from __future__ import annotations

from itertools import combinations

GRID_MM = 0.000001
GROUND_SETBACK_MM = 0.025
GROUND_RING_WIDTH_MM = 0.015
GROUND_SITE_SIZE_MM = 0.040
GROUND_SITE_OFFSET_MM = 0.135
PAD_SETBACK_MM = 0.085
PAD_WIDTH_MM = 0.130
PAD_LENGTH_MM = 0.120
PAD_CLEARANCE_MM = 0.040
TAPER_LENGTH_MM = 0.115
TAPER_OVERLAP_MM = 0.020
DRIVE_WIDTH_MM = 0.040
FLUX_WIDTH_MM = 0.010
FLUX_GAP_MM = 0.006
FLUX_TIP_Y_MM = -0.217
FLUX_RETURN_Y_MM = -0.280
FLUX_RETURN_X_OFFSET_MM = 0.120
FLUX_RETURN_WIDTH_MM = 0.030
COUPLER_X_MM = {"c01": -2.0, "c12": 0.0, "c23": 2.0}


def _edge_launch(x, y_edge, inward, width, gap):
    from shapely.geometry import Polygon, box

    outer = PAD_SETBACK_MM
    inner = outer + PAD_LENGTH_MM
    taper_end = inner + TAPER_LENGTH_MM
    connection = taper_end + TAPER_OVERLAP_MM

    def profile(sections):
        left = [(x - half_width, y_edge + inward * depth) for depth, half_width in sections]
        right = [(x + half_width, y_edge + inward * depth) for depth, half_width in reversed(sections)]
        return Polygon(left + right)

    signal = profile([(outer, PAD_WIDTH_MM / 2), (inner, PAD_WIDTH_MM / 2),
                      (taper_end, width / 2), (connection, width / 2)])
    slot = profile([(outer - PAD_CLEARANCE_MM, PAD_WIDTH_MM / 2 + PAD_CLEARANCE_MM),
                    (inner, PAD_WIDTH_MM / 2 + PAD_CLEARANCE_MM),
                    (taper_end, width / 2 + gap), (connection + gap, width / 2 + gap)])
    y0, y1 = sorted((y_edge + inward * outer, y_edge + inward * inner))
    return signal, slot, box(x - PAD_WIDTH_MM / 2, y0, x + PAD_WIDTH_MM / 2, y1)


def _contact(label, kind, conductor, pad):
    x, y = (round(value, 9) for value in pad.centroid.coords[0])
    return {"label": label, "kind": kind, "conductor": conductor, "signal_pad_mm": [x, y],
            "signal_pad_bounds_mm": [round(value, 9) for value in pad.bounds],
            "ground_pad_size_mm": GROUND_SITE_SIZE_MM,
            "ground_bond_sites_mm": [[round(x + offset, 9), y]
                                     for offset in (-GROUND_SITE_OFFSET_MM, GROUND_SITE_OFFSET_MM)]}


def _require_connection(first, second, label):
    from shapely import set_precision

    if first.intersection(second).area <= GRID_MM**2:
        raise ValueError(f"Launch {label} does not meet its on-chip conductor")
    joined = set_precision(first.union(second), GRID_MM)
    if joined.geom_type != "Polygon" or not joined.is_valid:
        raise ValueError(f"Launch {label} is disconnected or invalid")
    return joined


def _flux_routes(south, metal, slots, contacts):
    from shapely import set_precision
    from shapely.geometry import LineString, box

    returns = {}
    for coupler, x in COUPLER_X_MM.items():
        label = f"flux_{coupler}"
        pad_trace, slot, pad = _edge_launch(x, south, 1, FLUX_WIDTH_MM, FLUX_GAP_MM)
        return_x = x + FLUX_RETURN_X_OFFSET_MM
        tip = [(x, FLUX_TIP_Y_MM), (return_x, FLUX_TIP_Y_MM)]
        centerline = LineString([(x, south + PAD_SETBACK_MM + PAD_LENGTH_MM + TAPER_LENGTH_MM),
                                 *tip, (return_x, FLUX_RETURN_Y_MM)])
        line = centerline.buffer(FLUX_WIDTH_MM / 2, cap_style="flat", join_style="mitre")
        half_width = FLUX_RETURN_WIDTH_MM / 2
        x0, x1 = return_x - half_width, return_x + half_width
        landing = box(x0, FLUX_RETURN_Y_MM, x1, FLUX_RETURN_Y_MM + 0.020)
        metal[label] = _require_connection(pad_trace, line.union(landing), label)

        # Stop the slot at the flat end of the return pad. The grounded and
        # bias polygons share that edge while their interiors remain disjoint.
        return_cut = box(x0, FLUX_RETURN_Y_MM - 2 * FLUX_GAP_MM, x1, FLUX_RETURN_Y_MM)
        keepout = metal[label].buffer(FLUX_GAP_MM, join_style="mitre").union(slot)
        slots.append(keepout.difference(return_cut))
        returns[label] = set_precision(LineString([(x0, FLUX_RETURN_Y_MM), (x1, FLUX_RETURN_Y_MM)]), GRID_MM)
        contact = _contact(label, "flux", label, pad)
        contact.update(coupler=coupler, ground_return_mm=[return_x, FLUX_RETURN_Y_MM],
                       ground_return_width_mm=FLUX_RETURN_WIDTH_MM, coupling_segment_mm=[list(p) for p in tip])
        contacts.append(contact)
    return returns


def add_launch_layout(design, metal, *, feedlines):
    """Add RF launches and grounded DC flux routes to the four-qubit top metal.

    ``feedlines`` maps each readout port label to the name of its on-chip
    conductor in ``metal``. Optional ``device_keepouts_wkt`` in ``design`` holds
    additional process-layer ground exclusions in millimetres.
    """
    from shapely import from_wkt, set_precision
    from shapely.geometry import box
    from shapely.ops import unary_union

    from ..chain import linear_chain_geometry

    x0, south, _, x1, north, _ = design["enclosure_mm"]
    width = design["readout_cpw"]["trace_width_mm"]
    gap = design["readout_cpw"]["ground_gap_mm"]
    metal = dict(metal)
    launch_slots, contacts = [], []
    for port in design["ports"]:
        label = port["label"]
        x, y, _ = port["reference_plane_mm"]
        if label.startswith("drive_") and port["direction"] == "+Y" and y == south:
            kind, conductor, inward, trace_width, trace_gap = "drive", label, 1, DRIVE_WIDTH_MM, PAD_CLEARANCE_MM
        elif label in feedlines and port["direction"] == "-Y" and y == north:
            kind, conductor, inward, trace_width, trace_gap = "readout", feedlines[label], -1, width, gap
        else:
            raise ValueError(f"Unsupported chip-edge launch {label}")
        signal, slot, pad = _edge_launch(x, y, inward, trace_width, trace_gap)
        trimmed_until = y + inward * PAD_SETBACK_MM
        trim_lo, trim_hi = sorted((y, trimmed_until))
        trim_box = box(x - trace_width / 2 - GRID_MM, trim_lo,
                       x + trace_width / 2 + GRID_MM, trim_hi)
        metal[conductor] = _require_connection(metal[conductor].difference(trim_box), signal, label)
        launch_slots.append(slot)
        contacts.append(_contact(label, kind, conductor, pad))

    readout_channels = design["readout_channels"]
    readout_names = set(feedlines.values()) | {channel["resonator"] for channel in readout_channels}
    readout_names.update(name for name in metal if name.startswith("readout_"))
    slots = [shape.buffer(gap if name in readout_names else PAD_CLEARANCE_MM, join_style="mitre")
             for name, shape in metal.items()]
    base, base_metal = linear_chain_geometry(4)
    for channel in readout_channels:
        pickup = base["terminal_mapping"][channel["qubit"]]["readout"]
        slots.append(base_metal[pickup].buffer(PAD_CLEARANCE_MM, join_style="mitre"))
    device_keepouts = unary_union([from_wkt(wkt) for wkt in design.get("device_keepouts_wkt", {}).values()])
    slots.append(device_keepouts)
    slots.extend(launch_slots)
    returns = _flux_routes(south, metal, slots, contacts)

    footprint = set_precision(box(x0 + GROUND_SETBACK_MM, south + GROUND_SETBACK_MM,
                                  x1 - GROUND_SETBACK_MM, north - GROUND_SETBACK_MM), GRID_MM)
    ground = set_precision(footprint.difference(unary_union(slots)), GRID_MM)
    planes = ground.geoms if ground.geom_type == "MultiPolygon" else (ground,)
    ring = footprint.difference(footprint.buffer(-GROUND_RING_WIDTH_MM, join_style="mitre"))
    perimeter_ground = next((plane for plane in planes if plane.covers(ring)), None)
    if perimeter_ground is None:
        raise ValueError("Chip-edge ground ring must remain continuous around all launches")
    for label, shape in metal.items():
        intersection = shape.intersection(ground)
        if label in returns:
            if shape.intersects(device_keepouts):
                raise ValueError(f"Flux route {label} intersects a junction device exclusion")
            if not intersection.equals(returns[label]) or not perimeter_ground.covers(returns[label]):
                raise ValueError(f"Flux route {label} must meet ground at only its specified return edge")
        elif not intersection.is_empty:
            raise ValueError(f"Unexpected ground connection from {label}")
    for (label, shape), (other, counterpart) in combinations(metal.items(), 2):
        if shape.intersects(counterpart):
            raise ValueError(f"Signal conductors touch at {label} and {other}")
    for contact in contacts:
        pad = set_precision(box(*contact["signal_pad_bounds_mm"]), GRID_MM)
        if not metal[contact["conductor"]].covers(pad):
            raise ValueError(f"Signal bond site for {contact['label']} does not land on its conductor")
        half_width = contact["ground_pad_size_mm"] / 2
        for x, y in contact["ground_bond_sites_mm"]:
            if not perimeter_ground.covers(box(x - half_width, y - half_width, x + half_width, y + half_width)):
                raise ValueError(f"Ground bond site for {contact['label']} does not land on coplanar ground")
    return metal, contacts, ground
