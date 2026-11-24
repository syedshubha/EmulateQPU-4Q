"""Compose a reference multilayer chip layout from the four-qubit field geometry."""
from __future__ import annotations

from ..io import digest
from .design import four_qubit_design
from .layout import DATABASE_GRID_MM

HARDWARE_SCHEMA = "emulateqpu.fullwave.hardware.v1"
BRIDGE_HEIGHT_MM = 0.003


def _layer(number, sheets, *, fabrication=True, labels=None):
    layer = {"layer": number, "datatype": 0, "sheets": list(sheets), "fabrication": fabrication}
    if labels:
        layer["labels"] = labels
    return layer


def _bridge_sites(design, filters, contacts):
    for row in filters:
        for index, site in enumerate(row["feedline_bridge_sites_mm"]):
            yield f"bridge_{row['qubit']}_feed_{index}", row["feedline"], site["center_mm"], site["across"]
    for row in design.get("ground_bridges", ()):
        yield row["label"], "readout_feedline", row["center_mm"], "y"
    for contact in contacts:
        if contact["kind"] == "flux":
            x = contact["signal_pad_mm"][0]
            yield f"bridge_{contact['label']}", contact["conductor"], [x, -0.365], "x"


def _ground_bridges(design, filters, contacts, metal, ground):
    from shapely import set_precision
    from shapely.geometry import Point, box

    def rectangle(center, across, length, width):
        x, y = center
        dx, dy = (length / 2, width / 2) if across == "x" else (width / 2, length / 2)
        return set_precision(box(x - dx, y - dy, x + dx, y + dy), DATABASE_GRID_MM)

    ground_parts = tuple(ground.geoms) if ground.geom_type == "MultiPolygon" else (ground,)
    landings, straps, bridges, edges = {}, {}, [], []
    for name, conductor, center, across in _bridge_sites(design, filters, contacts):
        # A 60 µm center-to-center span fits the uniform 10/6 µm readout CPW.
        x, y = center
        dx, dy = (0.030, 0.0) if across == "x" else (0.0, 0.030)
        ends = [(x - dx, y - dy), (x + dx, y + dy)]
        strap = rectangle(center, across, 0.070, 0.010)
        crossed = {key for key, sheet in metal.items() if strap.intersection(sheet).area > 1e-10}
        if crossed != {conductor} or any(strap.intersects(other) for other in straps.values()):
            raise ValueError(f"Ground bridge {name} crosses unexpected metal: {sorted(crossed)}")
        feet, connection = [], []
        for index, end in enumerate(ends):
            key = f"{name}_contact_{index}"
            landing = rectangle(end, across, 0.016, 0.016)
            plane_ids = [i for i, part in enumerate(ground_parts) if part.covers(landing)]
            if len(plane_ids) != 1 or landing.intersection(strap).area <= 0:
                raise ValueError(f"Ground bridge contact {key} must land fully on coplanar ground")
            landings[key] = landing
            feet.append(key)
            connection.append(plane_ids[0])
        straps[name] = strap
        edges.append(connection)
        bridges.append({"label": name, "crosses": conductor, "center_mm": list(center), "span_axis": across,
                        "strap": name, "contacts": feet, "clearance_mm": BRIDGE_HEIGHT_MM})

    bonded = {i for i, part in enumerate(ground_parts) for row in contacts
              for x, y in row["ground_bond_sites_mm"] if part.covers(Point(x, y))}
    while True:
        reachable = bonded | {i for a, b in edges if a in bonded or b in bonded for i in (a, b)}
        if reachable == bonded:
            break
        bonded = reachable
    if len(bonded) != len(ground_parts):
        raise ValueError("A coplanar ground region lacks a connection to the package ground")
    return landings, straps, bridges


def _junction_bandages(metal, junction_sheets, junctions):
    from shapely import set_precision
    from shapely.geometry import box

    bandages = {}
    electrode_names = [row[end] for row in junctions for end in ("bottom_electrode", "top_electrode")]
    for row in junctions:
        x, y = row["center_mm"]
        contacts = []
        for side, key, island in zip((-1, 1), ("bottom_electrode", "top_electrode"), row["islands"]):
            x0, x1 = sorted((x + side * 0.0018, x + side * 0.0040))
            bandage = set_precision(box(x0, y - 0.00065, x1, y + 0.00065), DATABASE_GRID_MM)
            electrode = row[key]
            if ({name for name, shape in metal.items() if bandage.intersects(shape)} != {island}
                    or {name for name in electrode_names if bandage.intersects(junction_sheets[name])} != {electrode}
                    or bandage.intersection(metal[island]).area <= 0
                    or bandage.intersection(junction_sheets[electrode]).area <= 0
                    or bandage.intersects(junction_sheets[row["barrier"]])):
                raise ValueError(f"Junction contact for {electrode} does not make the intended bandaged connection")
            name = f"bandage_{electrode}"
            bandages[name] = bandage
            contacts.append({"layer_shape": name, "base_island": island, "electrode": electrode})
        row["base_contacts"] = contacts
    return bandages


def reference_hardware_design(*, readout_topology="dedicated"):
    """Return the layer geometry and interfaces for an assumed Al-on-silicon process."""
    from shapely.geometry import box

    from .filter_layout import filtered_readout_layout
    from .junction_layout import CRITICAL_CURRENT_DENSITY_A_M2, SPECIFIC_CAPACITANCE_FF_UM2, add_junction_layout
    from .launch_layout import add_launch_layout

    source, source_sheets = four_qubit_design(readout_topology=readout_topology)
    metal = {name: source_sheets[name] for name in source["conductors"] if name != "chip_ground"}
    metal, filters = filtered_readout_layout(source, metal)
    junction_sheets, junctions, squid_loops = add_junction_layout(source, metal)
    bandages = _junction_bandages(metal, junction_sheets, junctions)
    feedlines = {row[end]: row["feedline"] for row in source["readout_channels"]
                 for end in ("input_port", "output_port")}
    junction_layers = {**junction_sheets, **bandages}
    geometry_inputs = {**source, "device_keepouts_wkt": {key: shape.wkt for key, shape in junction_layers.items()}}
    metal, contacts, ground = add_launch_layout(geometry_inputs, metal, feedlines=feedlines)
    if any(ground.intersects(shape) for shape in junction_layers.values()):
        raise ValueError("Junction geometry touches the reference layout ground")
    landings, straps, bridges = _ground_bridges(source, filters, contacts, metal, ground)
    metal["chip_ground"] = ground

    x0, y0, z0, x1, y1, _ = source["enclosure_mm"]
    die = {"layout_chip_outline": box(x0, y0, x1, y1)}
    markers = {f"bond_{row['label']}": box(*row["signal_pad_bounds_mm"]) for row in contacts}
    ground_markers = {}
    for row in contacts:
        half_size = row["ground_pad_size_mm"] / 2
        for index, (x, y) in enumerate(row["ground_bond_sites_mm"]):
            ground_markers[f"ground_bond_{row['label']}_{index}"] = box(
                x - half_size, y - half_size, x + half_size, y + half_size)
    loop_markers = {row["aperture"]: row["coupler"] for row in squid_loops}
    barrier_markers = {row["barrier"]: f"JJ {row['label']}" for row in junctions}
    layers = {
        "al_base_metal": _layer(1, metal),
        "al_junction_bottom": _layer(2, [row["bottom_electrode"] for row in junctions]),
        "al_junction_top": _layer(3, [row["top_electrode"] for row in junctions]),
        "bridge_contact_opening": _layer(4, landings),
        "al_bridge_metal": _layer(5, straps),
        "al_junction_bandage": _layer(6, bandages),
        "chip_outline": _layer(100, die, fabrication=False),
        "signal_bond_site": _layer(101, markers, fabrication=False,
                                   labels={f"bond_{row['label']}": row["label"] for row in contacts}),
        "junction_overlap": _layer(102, barrier_markers, fabrication=False, labels=barrier_markers),
        "squid_aperture": _layer(103, loop_markers, fabrication=False, labels=loop_markers),
        "ground_bond_site": _layer(104, ground_markers, fabrication=False),
    }
    sheets = {**metal, **junction_layers, **landings, **straps, **die, **markers, **ground_markers}
    return {
        "schema": HARDWARE_SCHEMA, "design_id": f"four-data-line-{readout_topology}-hardware-v1",
        "gds_top_cell": f"QPU4_{readout_topology.upper()}_DIE_R1",
        "field_prototype_design_sha256": digest(source), "readout_topology": readout_topology,
        "units": "mm", "enclosure_mm": source["enclosure_mm"], "mode_order": source["mode_order"],
        "island_order": source["island_order"], "readout_channels": source["readout_channels"],
        "junction_devices": junctions, "squid_loops": squid_loops, "purcell_filters": filters,
        "signal_contacts": contacts, "ground_bridges": bridges, "layout_layers": layers,
        "sheets_wkt": {name: sheet.wkt for name, sheet in sheets.items()},
        "assumed_process": {"substrate": "high-resistivity silicon", "substrate_thickness_mm": -z0,
                            "substrate_relative_permittivity": source["substrate_permittivity"],
                            "base_metal": "aluminum", "junction_stack": "Al/AlOx/Al",
                            "junction_critical_current_density_A_per_cm2": CRITICAL_CURRENT_DENSITY_A_M2 / 1e4,
                            "junction_specific_capacitance_fF_per_um2": SPECIFIC_CAPACITANCE_FF_UM2,
                            "junction_base_contacts": "cleaned aluminum bandage landings outside each barrier",
                            "bridge_metal": "aluminum", "nominal_bridge_clearance_mm": BRIDGE_HEIGHT_MM,
                            "package_ground": "common ground with wire bonds at the indicated sites"},
        "scope": "Reference chip geometry and wire-bond interface with nominal process assumptions",
        "foundry_drc_qualified": False, "field_qualified": False,
    }
