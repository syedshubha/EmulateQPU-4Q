"""Matched planar routing for the candidate bandpass Purcell filters."""
from __future__ import annotations

import math

from scipy.constants import c

from .design import DESIGN_ID, DESIGN_SCHEMA, SHARED_DESIGN_ID


def _filter_centerline(pickup_x, length_mm):
    from shapely.geometry import LineString

    def route(turn_y):
        points = [(0.29, 2.65), (0.29, 3.18), (0.22, 3.18), (0.22, 3.74),
                  (-0.20, 3.74), (-0.20, turn_y)]
        for index, x in enumerate((-0.14, -0.08, -0.02, 0.04, 0.10, 0.16), start=1):
            points.extend(((x, points[-1][1]), (x, 3.78 if index % 2 else turn_y)))
        points.extend(((0.23, turn_y), (0.23, 4.80), (0.29, 4.80), (0.29, 4.50)))
        return LineString([(pickup_x + x, y) for x, y in points])

    # Seven vertical runs lengthen as the top rises and the final lead shortens.
    turn_y = 4.50 + (length_mm - route(4.50).length) / 6
    if not 4.50 < turn_y < 4.80:
        raise ValueError("The Purcell filter length does not fit its routing lane")
    return route(turn_y)


def _trace(path, width):
    from shapely import set_precision

    if not path.is_simple:
        raise ValueError("A filtered readout centerline crosses itself")
    return set_precision(path.buffer(width / 2, cap_style="flat", join_style="mitre"), 1e-6)


def filtered_readout_layout(design, metal):
    """Return copied signal metal with four filters and their geometry in millimeters."""
    from shapely.geometry import LineString

    if design["schema"] != DESIGN_SCHEMA or design["design_id"] not in (DESIGN_ID, SHARED_DESIGN_ID):
        raise ValueError("Purcell-filter routing requires a matched v2 readout layout")
    shared = design["design_id"] == SHARED_DESIGN_ID
    channels = design["readout_channels"]
    width = design["readout_cpw"]["trace_width_mm"]
    coupling_gap = design["readout_cpw"]["coupling_gap_mm"]
    effective_permittivity = (1 + design["substrate_permittivity"]) / 2
    west, _, _, east, north, _ = design["enclosure_mm"]
    trace_end, join_y = north - 0.05, north - 0.25
    updated, filter_rows, shared_path = dict(metal), [], []
    for index, row in enumerate(channels):
        resonator = metal[row["resonator"]]
        pickup_sample_y = resonator.bounds[1] + 0.02
        pickup_slice = LineString([(west, pickup_sample_y), (east, pickup_sample_y)])
        pickup_x = round(resonator.intersection(pickup_slice).centroid.x, 6)
        launch_x, coupled_x, far_x, right_x = (pickup_x + offset for offset in (0.29, 0.31, 0.37, 0.68))
        left_end = trace_end if not shared or index == 0 else join_y
        right_end = trace_end if not shared or index == len(channels) - 1 else join_y
        feed_path = [(launch_x, left_end), (launch_x, 4.90), (coupled_x, 4.90), (coupled_x, 3.52),
                     (far_x, 3.52), (far_x, 2.70), (right_x, 2.70), (right_x, right_end)]
        if shared:
            shared_path.extend(feed_path)
        else:
            updated[row["feedline"]] = _trace(LineString(feed_path), width)

        frequency = row["nominal_frequency_GHz"]
        length = c / (2 * frequency * 1e6 * math.sqrt(effective_permittivity))
        centerline = _filter_centerline(pickup_x, length)
        name = f"readout_{row['qubit']}_purcell_filter"
        updated[name] = _trace(centerline, width)
        bridge_sites = [(far_x, 3.35, "x"), (coupled_x, 4.18, "x"), (right_x, 3.25, "x"),
                        (right_x, 4.20, "x"), ((far_x + right_x) / 2, 2.70, "y")]
        filter_rows.append({"qubit": row["qubit"], "filter": name, "resonator": row["resonator"],
                            "feedline": row["feedline"], "centerline_length_mm": centerline.length,
                            "nominal_frequency_GHz": frequency, "trace_width_mm": width,
                            "resonator_coupling_length_mm": 0.45, "feedline_coupling_length_mm": 0.30,
                            "feedline_bridge_sites_mm": [
                                {"center_mm": [round(x, 6), y], "across": across} for x, y, across in bridge_sites]})
    if shared:
        updated[channels[0]["feedline"]] = _trace(LineString(shared_path), width)

    for row in filter_rows:
        resonator, filter_metal, feedline = (updated[row[key]] for key in ("resonator", "filter", "feedline"))
        resonator_gap = resonator.distance(filter_metal)
        feedline_gap = filter_metal.distance(feedline)
        direct_gap = resonator.distance(feedline)
        if (not math.isclose(resonator_gap, coupling_gap, abs_tol=1e-6)
                or not math.isclose(feedline_gap, coupling_gap, abs_tol=1e-6)
                or direct_gap < 0.08):
            raise ValueError(f"Invalid Purcell-filter coupling gaps for {row['qubit']}")
        row.update(resonator_filter_gap_mm=round(resonator_gap, 6), filter_feedline_gap_mm=round(feedline_gap, 6),
                   resonator_feedline_min_gap_mm=round(direct_gap, 6))
    return updated, filter_rows
