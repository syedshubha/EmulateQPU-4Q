"""Nominal coplanar chip carrier for the shared and dedicated four-qubit dies."""
from __future__ import annotations

from itertools import combinations
from math import pi, sqrt

from scipy.optimize import brentq
from scipy.special import ellipk

from ..io import digest
from .layout import DATABASE_GRID_MM

CAVITY_CLEARANCE_MM = 0.075
ALUMINA_EPSILON_R = 9.8
ALUMINA_THICKNESS_MM = 0.635
ATTACHMENT_THICKNESS_MM = 0.020
TRACE_WIDTH_MM = 0.080
PAD_GAP_MM = 0.040
PAD_SETBACK_MM = 0.075
TAPER_LENGTH_MM = 0.225
BOND_LOOP_HEIGHT_MM = 0.100
SHARED_ALIASES = {"readout_in": "readout_q0_in", "readout_out": "readout_q3_out"}


def _cpw_impedance(gap):
    k = TRACE_WIDTH_MM / (TRACE_WIDTH_MM + 2 * gap)
    epsilon_eff = (1 + ALUMINA_EPSILON_R) / 2
    return float(30 * pi / sqrt(epsilon_eff) * ellipk(1 - k**2) / ellipk(k**2))


def _carrier_route(chip_site, cavity_bounds, carrier_bounds, gap):
    from shapely import set_precision
    from shapely.geometry import Polygon, box

    label = chip_site["label"]
    x, chip_y = chip_site["signal_pad_mm"]
    south = chip_site["kind"] != "readout"
    sign = -1 if south else 1
    y_edge = cavity_bounds[1 if south else 3]
    y_outer = carrier_bounds[1 if south else 3]
    x0, y0, x1, y1 = chip_site["signal_pad_bounds_mm"]
    half_width = (x1 - x0) / 2
    pad_length = y1 - y0
    pad_end = PAD_SETBACK_MM + pad_length
    taper_end = pad_end + TAPER_LENGTH_MM
    outer = abs(y_outer - y_edge)
    mid_y = (cavity_bounds[1] + cavity_bounds[3]) / 2
    near_edge = y_edge < chip_y < mid_y if south else mid_y < chip_y < y_edge
    grounds = chip_site["ground_bond_sites_mm"]
    if (not near_edge or taper_end >= outer or not (x0 < x < x1 and y0 < chip_y < y1)
            or len(grounds) != 2 or not grounds[0][0] < x < grounds[1][0]
            or any(gy != chip_y for _, gy in grounds)):
        raise ValueError(f"Carrier site {label} is not aligned to its die edge")

    def profile(sections):
        left = [(x - half, y_edge + sign * depth) for depth, half in sections]
        right = [(x + half, y_edge + sign * depth) for depth, half in reversed(sections)]
        return set_precision(Polygon(left + right), DATABASE_GRID_MM)

    route = profile([(PAD_SETBACK_MM, half_width), (pad_end, half_width),
                     (taper_end, TRACE_WIDTH_MM / 2), (outer, TRACE_WIDTH_MM / 2)])
    slot = profile([(PAD_SETBACK_MM - PAD_GAP_MM, half_width + PAD_GAP_MM),
                    (pad_end, half_width + PAD_GAP_MM), (taper_end, TRACE_WIDTH_MM / 2 + gap),
                    (outer + gap, TRACE_WIDTH_MM / 2 + gap)])
    near, far = sorted((y_edge + sign * PAD_SETBACK_MM, y_edge + sign * pad_end))
    signal_pad = set_precision(box(x - half_width, near, x + half_width, far), DATABASE_GRID_MM)
    y = round((near + far) / 2, 9)
    ground_sites = [[gx, y] for gx, _ in grounds]
    landing_half = chip_site["ground_pad_size_mm"] / 2
    ground_pads = [set_precision(box(gx - landing_half, gy - landing_half,
                                     gx + landing_half, gy + landing_half), DATABASE_GRID_MM)
                   for gx, gy in ground_sites]
    site = {"label": label, "kind": chip_site["kind"], "conductor": f"carrier_signal_{label}",
            "signal_pad_mm": [x, y], "signal_pad_bounds_mm": list(signal_pad.bounds),
            "ground_bond_sites_mm": ground_sites,
            "ground_pad_size_mm": chip_site["ground_pad_size_mm"], "outer_reference_plane_mm": [x, y_outer, 0.0]}
    return site, route, slot, signal_pad, ground_pads


def _wire_bond_assembly(chip, dedicated, carrier_sites, aliases):
    def wire(start, end):
        x, y = start
        if end[0] != x or y == end[1]:
            raise ValueError("Carrier wire bonds must be aligned and span the die-to-carrier gap")
        return [[x, y, 0.0], [x, round((y + end[1]) / 2, 9), BOND_LOOP_HEIGHT_MM], [*end, 0.0]]

    bonded, used = [], set()
    for site in chip["signal_contacts"]:
        label = site["label"]
        target = aliases.get(label, label)
        reference = dedicated[target]
        carrier = carrier_sites[target]
        if (target in used or site["kind"] != reference["kind"]
                or any(site[key] != reference[key] for key in ("signal_pad_mm", "signal_pad_bounds_mm",
                                                               "ground_bond_sites_mm", "ground_pad_size_mm"))):
            raise ValueError(f"Chip contact {label} does not fit reference carrier site {target}")
        used.add(target)
        bonded.append({"die_site": label, "carrier_site": target,
                       "signal_wire_mm": wire(site["signal_pad_mm"], carrier["signal_pad_mm"]),
                       "ground_wires_mm": [wire(start, stop) for start, stop in
                                           zip(site["ground_bond_sites_mm"], carrier["ground_bond_sites_mm"],
                                               strict=True)]})
    unbonded = [name for name in carrier_sites if name not in used]
    return {"chip_design_id": chip["design_id"], "chip_design_sha256": digest(chip),
            "bond_loop_height_mm": BOND_LOOP_HEIGHT_MM, "nominal_bond_diameter_mm": 0.025,
            "bond_material": "aluminum", "bonded": bonded, "unbonded_carrier_sites": unbonded,
            "unbonded_outer_loads_ohm": {name: 50.0 for name in unbonded}}


def reference_carrier_design(dedicated_hardware, shared_hardware):
    """Return one nominal alumina carrier pattern and both wire-bond assemblies."""
    from shapely import set_precision
    from shapely.geometry import box
    from shapely.ops import unary_union

    dedicated, shared = dedicated_hardware, shared_hardware
    if (dedicated["readout_topology"] != "dedicated" or shared["readout_topology"] != "shared"
            or dedicated["enclosure_mm"] != shared["enclosure_mm"]):
        raise ValueError("Carrier requires matched dedicated and shared die outlines")
    contacts = {row["label"]: row for row in dedicated["signal_contacts"]}
    if len(contacts) != 15 or len(dedicated["signal_contacts"]) != 15 or len(shared["signal_contacts"]) != 9:
        raise ValueError("Expected 15 dedicated and nine shared GSG sites")
    die_thickness = dedicated["assumed_process"]["substrate_thickness_mm"]
    depth = die_thickness + ATTACHMENT_THICKNESS_MM
    if die_thickness != shared["assumed_process"]["substrate_thickness_mm"] or not 0 < depth < ALUMINA_THICKNESS_MM:
        raise ValueError("Chip substrate and attach film must fit a common recessed carrier")
    x0, south, _, x1, north, _ = dedicated["enclosure_mm"]
    cavity = set_precision(box(x0 - CAVITY_CLEARANCE_MM, south - CAVITY_CLEARANCE_MM,
                                x1 + CAVITY_CLEARANCE_MM, north + CAVITY_CLEARANCE_MM), DATABASE_GRID_MM)
    outline = set_precision(box(x0 - 0.850, cavity.bounds[1] - 1.750,
                                 x1 + 0.850, cavity.bounds[3] + 1.750), DATABASE_GRID_MM)
    gap = round(brentq(lambda value: _cpw_impedance(value) - 50., 0.001, 0.250), 6)

    sites, routes, signal_pads, ground_pads = {}, {}, {}, {}
    slots = [cavity]
    for label, chip_site in contacts.items():
        site, route, slot, signal_pad, ground_landings = _carrier_route(chip_site, cavity.bounds, outline.bounds, gap)
        sites[label], routes[site["conductor"]] = site, route
        signal_pads[f"carrier_bond_{label}"] = signal_pad
        ground_pads.update({f"carrier_ground_bond_{label}_{i}": pad for i, pad in enumerate(ground_landings)})
        slots.extend((slot, route.buffer(gap, join_style="mitre")))
        if not route.covers(signal_pad) or route.intersects(cavity):
            raise ValueError(f"Carrier route {label} misses its bond pad or enters the cavity")

    ground = set_precision(outline.difference(unary_union(slots)), DATABASE_GRID_MM)
    if ground.geom_type != "Polygon" or not ground.is_valid:
        raise ValueError("Carrier ground must be a single connected conductor")
    if (any(ground.intersects(route) for route in routes.values())
            or any(a.intersects(b) for a, b in combinations(routes.values(), 2))
            or any(not ground.covers(pad) for pad in ground_pads.values())):
        raise ValueError("Carrier routes or ground bond landings have an unintended connection or gap")

    assemblies = {"dedicated": _wire_bond_assembly(dedicated, contacts, sites, {}),
                  "shared": _wire_bond_assembly(shared, contacts, sites, SHARED_ALIASES)}
    if len(assemblies["shared"]["unbonded_carrier_sites"]) != 6:
        raise ValueError("The shared die must leave six dedicated carrier routes unbonded")
    sheets = {**routes, "carrier_ground": ground, "carrier_outline": outline, "carrier_die_pocket": cavity,
              **signal_pads, **ground_pads}

    def layer(number, names, *, fabrication, labels=None):
        result = {"layer": number, "datatype": 0, "sheets": list(names), "fabrication": fabrication}
        if labels:
            result["labels"] = labels
        return result

    layers = {"carrier_top_metal": layer(1, [*routes, "carrier_ground"], fabrication=True),
              "die_pocket_removal": layer(20, ["carrier_die_pocket"], fabrication=True),
              "carrier_outline": layer(100, ["carrier_outline"], fabrication=False),
              "carrier_signal_bonds": layer(101, signal_pads, fabrication=False,
                                            labels={f"carrier_bond_{label}": label for label in sites}),
              "carrier_ground_bonds": layer(102, ground_pads, fabrication=False)}
    xmin, ymin, xmax, ymax = outline.bounds
    return {"schema": "emulateqpu.fullwave.carrier.v1", "design_id": "four-data-line-reference-carrier-v1",
            "gds_top_cell": "QPU4_COMMON_CARRIER_R1",
            "units": "mm", "enclosure_mm": [xmin, ymin, -ALUMINA_THICKNESS_MM, xmax, ymax, BOND_LOOP_HEIGHT_MM],
            "carrier_sites": list(sites.values()), "wire_bond_assemblies": assemblies,
            "assumed_stack": {"substrate": "alumina", "substrate_relative_permittivity": ALUMINA_EPSILON_R,
                              "substrate_thickness_mm": ALUMINA_THICKNESS_MM,
                              "pocket_side_clearance_mm": CAVITY_CLEARANCE_MM, "pocket_depth_mm": depth,
                              "attach_film_thickness_mm": ATTACHMENT_THICKNESS_MM,
                              "die_top_plane_z_mm": 0.0, "carrier_top_plane_z_mm": 0.0},
            "cpw": {"center_width_mm": TRACE_WIDTH_MM, "gap_mm": gap, "estimated_Z0_ohm": _cpw_impedance(gap),
                    "model": "Zero-thickness CPW, quasi-static half-space elliptic-integral estimate"},
            "layout_layers": layers, "sheets_wkt": {name: geometry.wkt for name, geometry in sheets.items()},
            "foundry_drc_qualified": False, "package_em_qualified": False,
            "scope": "Nominal common carrier geometry and wire-bond assemblies"}
