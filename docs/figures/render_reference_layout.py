"""Render the two reference dies from the same CAD used for GDSII export."""

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

from shapely import from_wkt

from emulateqpu.fullwave.hardware import reference_hardware_design


WIDTH, HEIGHT = 1080, 664
PANEL_WIDTH, PANEL_TOP = 502, 135
INK = "#152d47"
MUTED = "#405773"
STYLES = {
    "island": ("Qubit and coupler islands", INK, 0),
    "feedline": ("Readout feedline", "#6742b5", 1.2),
    "resonator": ("Readout resonator", "#00727c", 0.8),
    "filter": ("Purcell filter", "#a75208", 0.8),
    "drive": ("Microwave drive", "#2866c2", 0.5),
    "flux": ("Coupler flux line", "#aa3b68", 0.5),
}


def add_text(parent, x, y, label, *, size=19, color=MUTED, weight=400, anchor="start"):
    element = ET.SubElement(parent, "text", {
        "x": f"{x:.2f}", "y": f"{y:.2f}", "fill": color, "font-size": str(size),
        "font-weight": str(weight), "text-anchor": anchor,
    })
    element.text = label


def polygon_path(geometry):
    polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else (geometry,)
    commands = []
    for polygon in polygons:
        for ring in (polygon.exterior, *polygon.interiors):
            points = " ".join(f"{x:.6f},{y:.6f}" for x, y in ring.coords[:-1])
            commands.append(f"M {points} Z")
    return " ".join(commands)


def draw_panel(svg, design, *, x, scale, title):
    x_min, y_min, _, x_max, y_max, _ = design["enclosure_mm"]
    panel_height = scale * (y_max - y_min)
    feedlines = list(dict.fromkeys(channel["feedline"] for channel in design["readout_channels"]))
    contacts = design["signal_contacts"]
    sheets_by_role = {
        "island": design["island_order"],
        "feedline": feedlines,
        "resonator": [channel["resonator"] for channel in design["readout_channels"]],
        "filter": [row["filter"] for row in design["purcell_filters"]],
        "drive": [row["conductor"] for row in contacts if row["kind"] == "drive"],
        "flux": [row["conductor"] for row in contacts if row["kind"] == "flux"],
    }
    add_text(svg, x, 87, title, size=24, color=INK, weight=700)
    feedline_label = "feedline" if len(feedlines) == 1 else "feedlines"
    add_text(svg, x, 113, f"{len(feedlines)} {feedline_label} · {len(contacts)} signal pads")

    ET.SubElement(svg, "rect", {
        "x": str(x), "y": str(PANEL_TOP), "width": f"{scale * (x_max - x_min):.2f}",
        "height": f"{panel_height:.2f}", "fill": "#f2f5f8", "stroke": "#a5b4c5",
    })
    metal = ET.SubElement(svg, "g", {
        "transform": f"translate({x - x_min * scale} {PANEL_TOP + y_max * scale}) scale({scale} {-scale})",
        "fill-rule": "evenodd",
    })
    for role, sheet_names in sheets_by_role.items():
        _, color, stroke_width = STYLES[role]
        for name in sheet_names:
            shape = from_wkt(design["sheets_wkt"][name])
            ET.SubElement(metal, "path", {
                "d": polygon_path(shape), "fill": color, "stroke": color,
                "stroke-width": str(stroke_width), "stroke-linejoin": "round",
                "vector-effect": "non-scaling-stroke",
            })

    mode_x = {row["mode"]: row["center_mm"][0] for row in design["junction_devices"]}
    for mode, center in mode_x.items():
        add_text(svg, x + (center - x_min) * scale, PANEL_TOP + panel_height + 25,
                 mode, color=INK, weight=700, anchor="middle")


def draw_legend(svg, scale):
    columns = (29, 294, 558)
    for index, (label, color, _) in enumerate(STYLES.values()):
        x = columns[index % len(columns)]
        y = 587 + 28 * (index // len(columns))
        ET.SubElement(svg, "rect", {
            "x": str(x), "y": str(y - 10), "width": "20", "height": "5", "rx": "2.5", "fill": color,
        })
        add_text(svg, x + 29, y, label)

    right = WIDTH - 28
    left = right - scale
    ET.SubElement(svg, "path", {
        "d": f"M{left:.2f},583 v9 M{left:.2f},587.5 H{right} M{right},583 v9",
        "fill": "none", "stroke": INK, "stroke-width": "1.5",
    })
    add_text(svg, (left + right) / 2, 615, "1 mm", color=INK, anchor="middle")
    add_text(svg, 29, 646, "Ground plane omitted. Trace widths emphasized.", size=17)


def render(output):
    dedicated = reference_hardware_design(readout_topology="dedicated")
    shared = reference_hardware_design(readout_topology="shared")
    x_min, _, _, x_max, _, _ = dedicated["enclosure_mm"]
    scale = PANEL_WIDTH / (x_max - x_min)
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "viewBox": f"0 0 {WIDTH} {HEIGHT}",
        "width": str(WIDTH), "height": str(HEIGHT), "role": "img",
        "aria-labelledby": "title description", "font-family": "Arial, sans-serif",
    })
    ET.SubElement(svg, "title", {"id": "title"}).text = "Four-qubit reference die layouts"
    ET.SubElement(svg, "desc", {"id": "description"}).text = (
        "Top views of generated reference CAD. Four dedicated readout feedlines use 15 signal pads. "
        "One continuous shared feedline uses nine signal pads. Both layouts use the same four data "
        "transmons, three tunable couplers, readout resonators and Purcell filters."
    )
    ET.SubElement(svg, "rect", {"width": "100%", "height": "100%", "fill": "white"})
    add_text(svg, 28, 43, "Four-qubit reference die layouts", size=32, color=INK, weight=700)
    draw_panel(svg, dedicated, x=28, scale=scale, title="Dedicated readout")
    draw_panel(svg, shared, x=550, scale=scale, title="Shared readout")
    draw_legend(svg, scale)
    ET.indent(svg)
    output.write_text(ET.tostring(svg, encoding="unicode") + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path(__file__).with_name("reference-die-layouts.svg"))
    render(parser.parse_args().output)
