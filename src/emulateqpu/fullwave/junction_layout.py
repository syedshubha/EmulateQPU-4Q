"""Draw nominal aluminum junctions and SQUID loops for the four-qubit layout."""
from __future__ import annotations

import math

from scipy.constants import elementary_charge

from .design import MODE_ORDER
from .layout import DATABASE_GRID_MM

CRITICAL_CURRENT_DENSITY_A_M2 = 3e5  # 30 A/cm²
SPECIFIC_CAPACITANCE_FF_UM2 = 50.0
COUPLER_Y_MM = -0.175
SQUID_BRANCH_OFFSET_MM = 0.006


def _taper(x0, x1, y, width0, width1):
    from shapely.geometry import Polygon

    return Polygon([(x0, y - width0 / 2), (x1, y - width1 / 2),
                    (x1, y + width1 / 2), (x0, y + width0 / 2)])


def _base_lead(x, edge, side, y, squid):
    from shapely import set_precision
    from shapely.ops import unary_union

    inside = edge + side * 0.002
    tip = x + side * 0.0015
    if squid:
        trunk = _taper(inside, x + side * 0.0075, y, 0.016, 0.015)
        fingers = [_taper(x + side * 0.0085, tip, y + offset, 0.002, 0.0012)
                   for offset in (-SQUID_BRANCH_OFFSET_MM, SQUID_BRANCH_OFFSET_MM)]
        lead = unary_union([trunk, *fingers])
    else:
        lead = _taper(inside, tip, y, 0.008, 0.0012)
    return set_precision(lead, DATABASE_GRID_MM)


def _evaporated_electrode(x, y, side, half_overlap):
    from shapely import set_precision
    from shapely.geometry import Polygon

    far_end = x + side * 0.0026
    shoulder = x + side * 0.00055
    tip = x - side * half_overlap
    shape = Polygon([(far_end, y - 0.00045), (shoulder, y - half_overlap),
                     (tip, y - half_overlap), (tip, y + half_overlap),
                     (shoulder, y + half_overlap), (far_end, y + 0.00045)])
    return set_precision(shape, DATABASE_GRID_MM)


def _junction(sheets, label, mode, islands, x, y, target_ej):
    current_per_GHz = 4 * math.pi * elementary_charge * 1e9
    target_current = current_per_GHz * target_ej
    target_area_m2 = target_current / CRITICAL_CURRENT_DENSITY_A_M2
    half_overlap = (round(1e3 * math.sqrt(target_area_m2) / (2 * DATABASE_GRID_MM))
                    * DATABASE_GRID_MM)
    if not 0.0 < half_overlap < 0.00055:
        raise ValueError(f"Junction {label} falls outside the nominal electrode dimensions")
    bottom = f"jj_bottom_{label}"
    top = f"jj_top_{label}"
    barrier = f"jj_barrier_{label}"
    sheets[bottom] = _evaporated_electrode(x, y, -1, half_overlap)
    sheets[top] = _evaporated_electrode(x, y, 1, half_overlap)
    overlap = sheets[bottom].intersection(sheets[top])
    if overlap.geom_type != "Polygon" or overlap.area <= 0:
        raise ValueError(f"Junction {label} has no single tunnel overlap")
    sheets[barrier] = overlap
    area_um2 = round(overlap.area * 1e6, 6)
    current = area_um2 * 1e-12 * CRITICAL_CURRENT_DENSITY_A_M2
    return {"label": label, "mode": mode, "islands": list(islands),
            "bottom_electrode": bottom, "top_electrode": top, "barrier": barrier,
            "center_mm": [x, y], "critical_current_A": current,
            "target_critical_current_A": target_current,
            "junction_energy_GHz": current / current_per_GHz,
            "target_junction_energy_GHz": target_ej,
            "barrier_area_um2": area_um2, "capacitance_fF": area_um2 * SPECIFIC_CAPACITANCE_FF_UM2}


def _validate_contacts(conductors, island_names, sheets, junction_devices):
    for island in island_names:
        if any(island != name and conductors[island].intersects(shape)
               for name, shape in conductors.items()):
            raise ValueError(f"Base lead on {island} touches another conductor")
    electrodes = [device[key] for device in junction_devices
                  for key in ("bottom_electrode", "top_electrode")]
    for device in junction_devices:
        bottom, top = device["bottom_electrode"], device["top_electrode"]
        for name, other_name, island in zip((bottom, top), (top, bottom), device["islands"]):
            shape = sheets[name]
            if (shape.intersection(conductors[island]).area <= 0
                    or any(shape.intersects(conductor)
                           for key, conductor in conductors.items() if key != island)
                    or any(shape.intersects(sheets[key]) for key in electrodes if key not in (name, other_name))):
                raise ValueError(f"Junction electrode {name} has incorrect contacts")


def add_junction_layout(design, conductors):
    """Add tapered island leads and return layered junction and aperture polygons in mm.

    The design supplies the four-qubit mode order, paired island order and idle
    Josephson energies. This models matched SQUID junctions with negligible loop
    inductance. The overlap and aperture sheets are review markers. The two
    projected aluminum electrodes are separated by the tunnel barrier where
    they overlap. Electrical parameters use assumed process densities.
    """
    from shapely import set_precision
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union

    modes = design["mode_order"]
    island_names = design["island_order"]
    design_junctions = design["junctions"]
    if (tuple(modes) != MODE_ORDER
            or len(island_names) != 14 or len(set(island_names)) != 14
            or [row["label"] for row in design_junctions] != list(modes)
            or any(name not in conductors for name in island_names)):
        raise ValueError("Expected the ordered four-qubit design and its fourteen islands")
    updated = dict(conductors)
    sheets, junction_devices, squid_loops = {}, [], []
    for index, row in enumerate(design_junctions):
        mode = row["label"]
        left, right = island_names[2 * index:2 * index + 2]
        left_edge, right_edge = conductors[left].bounds[2], conductors[right].bounds[0]
        if not math.isclose(right_edge - left_edge, 0.030, abs_tol=DATABASE_GRID_MM):
            raise ValueError(f"Expected a 30 µm gap between the original islands for {mode}")
        x = round((left_edge + right_edge) / 2, 6)
        squid = mode.startswith("c")
        y = COUPLER_Y_MM if squid else 0.0
        for island, edge, side in ((left, left_edge, -1), (right, right_edge, 1)):
            lead = _base_lead(x, edge, side, y, squid)
            base = set_precision(conductors[island], DATABASE_GRID_MM)
            merged = base.union(lead)
            if lead.intersection(base).area <= 0 or merged.geom_type != "Polygon" or not merged.is_valid:
                raise ValueError(f"Invalid base lead on {island}")
            updated[island] = merged
        target_ej = row["EJ_GHz"]
        if squid:
            bias = row["bias_Phi0"]
            if not -0.5 < bias < 0.5:
                raise ValueError(f"Idle flux for {mode} must be inside the matched SQUID branch")
            idle_scale = math.cos(math.pi * bias)
            pair = [_junction(sheets, f"{mode}_{branch}", mode, (left, right), x, round(y + offset, 6),
                              target_ej / (2 * idle_scale))
                    for branch, offset in (("lower", -SQUID_BRANCH_OFFSET_MM), ("upper", SQUID_BRANCH_OFFSET_MM))]
            joined = unary_union([updated[left], updated[right],
                                  *(sheets[device[key]] for device in pair
                                    for key in ("bottom_electrode", "top_electrode"))])
            if joined.geom_type != "Polygon" or len(joined.interiors) != 1:
                raise ValueError(f"Expected a single aperture in SQUID {mode}")
            aperture = Polygon(joined.interiors[0])
            if (not aperture.contains(Point(x, y)) or
                    any(aperture.intersects(shape) for name, shape in updated.items() if name not in (left, right))):
                raise ValueError(f"SQUID aperture {mode} is obstructed")
            marker = f"squid_aperture_{mode}"
            sheets[marker] = aperture
            junction_devices.extend(pair)
            squid_loops.append({"coupler": mode, "junctions": [device["label"] for device in pair],
                                "center_mm": [x, y], "aperture": marker,
                                "aperture_area_um2": round(aperture.area * 1e6, 7),
                                "idle_bias_Phi0": bias, "effective_idle_EJ_GHz": target_ej,
                                "estimated_effective_idle_EJ_GHz": idle_scale * sum(
                                    device["junction_energy_GHz"] for device in pair),
                                "normal": "+Z"})
        else:
            junction_devices.append(_junction(sheets, mode, mode, (left, right), x, y, target_ej))
    _validate_contacts(updated, island_names, sheets, junction_devices)
    conductors.update({name: updated[name] for name in island_names})
    return sheets, junction_devices, squid_loops
