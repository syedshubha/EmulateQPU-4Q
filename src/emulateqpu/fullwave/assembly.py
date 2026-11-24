"""Prepare electromagnetic field inputs for the reference die on its common carrier."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ..io import atomic_json, digest
from .design import (ASSEMBLY_DESIGN_SCHEMA, ASSEMBLY_PORT_ORDER, READOUT_TOPOLOGIES,
                     _configs, _frequency_grids, _write_field_inputs)
from .hardware import HARDWARE_SCHEMA, reference_hardware_design
from .layout import DATABASE_GRID_MM

ASSEMBLY_SOURCES = ("chip-layout.json", "carrier-layout.json")
ATTACH_PERMITTIVITY = 3.2
FLUX_TABLES = ("surface-F.csv", "terminal-I.csv")


def _volume_bounds(chip, carrier):
    from shapely import from_wkt

    x0, y0, z0, x1, y1, _ = chip["enclosure_mm"]
    cx0, cy0, cz0, cx1, cy1, _ = carrier["enclosure_mm"]
    px0, py0, px1, py1 = from_wkt(carrier["sheets_wkt"]["carrier_die_pocket"]).bounds
    pocket_bottom = -carrier["assumed_stack"]["pocket_depth_mm"]
    if not (px0 < x0 < x1 < px1 and py0 < y0 < y1 < py1 and cz0 < pocket_bottom < z0 < 0):
        raise ValueError("The die and attachment layer do not fit the carrier pocket")
    return {
        "bounds": [cx0 - .30, cy0, cz0 - .25, cx1 + .30, cy1, .80],
        "die_bounds": [x0, y0, z0, x1, y1, 0.0],
        "carrier_bounds": [cx0, cy0, cz0, cx1, cy1, 0.0],
        "pocket_bounds": [px0, py0, pocket_bottom, px1, py1, 0.0],
        "attach_bounds": [x0, y0, pocket_bottom, x1, y1, z0],
    }


def _carrier_port_sheets(carrier, planar, assembly):
    from shapely import set_precision
    from shapely.geometry import box

    active = {row["carrier_site"]: row["die_site"] for row in assembly["bonded"]}
    sites = {row["label"]: row for row in carrier["carrier_sites"]}
    if set(sites) != set(ASSEMBLY_PORT_ORDER):
        raise ValueError("The common carrier must have the fifteen reference sites")
    half = carrier["cpw"]["center_width_mm"] / 2
    gap = carrier["cpw"]["gap_mm"]
    ground = planar["carrier_ground"]
    ports = []
    for number, label in enumerate(ASSEMBLY_PORT_ORDER, 1):
        site = sites[label]
        conductor = site["conductor"]
        x, edge, _ = site["outer_reference_plane_mm"]
        inward = -1 if site["kind"] == "readout" else 1
        end_y = edge + inward * .03
        y0, y1 = sorted((end_y, edge + inward * .15))
        strip_end = box(x - half - gap, min(end_y, edge), x + half + gap, max(end_y, edge))
        trace = set_precision(planar[conductor].difference(strip_end), DATABASE_GRID_MM)
        if trace.geom_type != "Polygon":
            raise ValueError(f"Carrier port {label} has a disconnected trace")
        planar[conductor] = trace
        parts = []
        for side, lo, hi, direction in (("left", x - half - gap, x - half, "+X"),
                                         ("right", x + half, x + half + gap, "-X")):
            surface = f"port_{label}_{side}"
            patch = set_precision(box(lo, y0, hi, y1), DATABASE_GRID_MM)
            for contact in (trace.intersection(patch), ground.intersection(patch)):
                if contact.area > 1e-12 or abs(contact.length - .12) > 1e-8:
                    raise ValueError(f"Carrier port patch {surface} does not meet the signal and ground edges")
            planar[surface] = patch
            parts.append({"surface": surface, "direction": direction})
        ports.append({"label": label, "port": number, "kind": site["kind"], "active": label in active,
                      "die_site": active.get(label), "Z_ref_ohm": 50.0, "elements": parts,
                      "reference_plane_mm": [x, (y0 + y1) / 2, 0.0]})
    return ports


def _bond_sheets(assembly, contacts, carrier_sites, planar):
    from shapely import set_precision
    from shapely.geometry import LineString

    half = assembly["nominal_bond_diameter_mm"] / 2
    shapes = {}
    for bond in assembly["bonded"]:
        name = bond["carrier_site"]
        chip_site = contacts[bond["die_site"]]
        carrier_site = carrier_sites[name]
        wires = (("signal", bond["signal_wire_mm"], chip_site["conductor"], carrier_site["conductor"]),
                 *((f"ground_{i}", line, "chip_ground", "carrier_ground")
                   for i, line in enumerate(bond["ground_wires_mm"])))
        for role, wire, die_conductor, carrier_conductor in wires:
            for (x, y, z), conductor in zip((wire[0], wire[-1]), (die_conductor, carrier_conductor)):
                foot = set_precision(LineString([(x - half, y), (x + half, y)]), DATABASE_GRID_MM)
                if z != 0 or not planar[conductor].covers(foot):
                    raise ValueError(f"Wire bond {name}/{role} does not land on {conductor}")
            shapes[f"bond_{name}_{role}"] = [
                [[xa - half, ya, za], [xa + half, ya, za], [xb + half, yb, zb], [xb - half, yb, zb]]
                for (xa, ya, za), (xb, yb, zb) in zip(wire, wire[1:])
            ]
    return shapes


def assembly_geometry(chip, carrier, *, flux_transfer=False):
    """Project both layout sources onto one conformal, fixed-bias assembly model."""
    from shapely import from_wkt

    from .chip_field import chip_field_geometry

    topology = chip["readout_topology"]
    if chip.get("schema") != HARDWARE_SCHEMA or topology not in READOUT_TOPOLOGIES:
        raise ValueError("Expected a dedicated or shared reference hardware die")
    if carrier.get("schema") != "emulateqpu.fullwave.carrier.v1":
        raise ValueError("Expected the common reference carrier")
    bonded = carrier["wire_bond_assemblies"][topology]
    if digest(chip) != bonded["chip_design_sha256"]:
        raise ValueError("The selected hardware die differs from the carrier wire-bond assembly")
    volumes = _volume_bounds(chip, carrier)
    planar, junctions, nonplanar = chip_field_geometry(chip)
    planar.update({name: from_wkt(carrier["sheets_wkt"][name])
                   for name in carrier["layout_layers"]["carrier_top_metal"]["sheets"]})
    nonplanar.update(_bond_sheets(bonded, {row["label"]: row for row in chip["signal_contacts"]},
                                 {row["label"]: row for row in carrier["carrier_sites"]}, planar))
    conductors = [name for name in planar if not name.startswith("port_jj_")] + list(nonplanar)
    ports = _carrier_port_sheets(carrier, planar, bonded)
    carrier_by_die = {row["die_site"]: row["carrier_site"] for row in bonded["bonded"]}
    channels = [{**row, "input_port": carrier_by_die[row["input_port"]],
                 "output_port": carrier_by_die[row["output_port"]]} for row in chip["readout_channels"]]
    bias_by_mode = {row["coupler"]: row["idle_bias_Phi0"] for row in chip["squid_loops"]}
    for row in junctions:
        row["EJ_GHz"] = row["linearized_EJ_GHz"]
        row["bias_Phi0"] = bias_by_mode.get(row["mode"], 0.0)
    design = {
        "schema": ASSEMBLY_DESIGN_SCHEMA, "design_id": f"four-data-line-{topology}-assembly-v1",
        "chip_design_sha256": digest(chip), "carrier_design_sha256": digest(carrier),
        "readout_topology": topology, "units": "mm", "enclosure_mm": volumes["bounds"],
        "volume_bounds_mm": volumes, "mode_order": chip["mode_order"], "island_order": chip["island_order"],
        "conductors": conductors, "ports": ports, "junctions": junctions, "readout_channels": channels,
        "sheets_wkt": {name: sheet.wkt for name, sheet in planar.items()},
        "raised_sheets_mm": nonplanar,
        "material_permittivity": {"silicon": chip["assumed_process"]["substrate_relative_permittivity"],
                                  "alumina": carrier["assumed_stack"]["substrate_relative_permittivity"],
                                  "attach": ATTACH_PERMITTIVITY, "air": 1.0},
        "assumptions": {
            "conductors": "PEC sheets with zero metal thickness and kinetic inductance",
            "bonds": "Diameter-wide PEC ribbons following the specified three-point bond centerlines",
            "junctions": "Ten lumped Josephson elements linearized at the prescribed idle loop fluxes",
            "ports": "Two transverse CPW-gap sheets per 50 ohm carrier port, including unbonded routes",
            "package": "Ideal PEC enclosure, fixed lid spacing, and nominal unmeasured attachment dielectric",
            "dielectric_loss_tangent": 0.0,
        },
    }
    if flux_transfer:
        from .flux_geometry import add_flux_apertures

        pickups = add_flux_apertures(chip, planar)
        design["flux_transfer"] = {
            "sources": [{"label": row["label"], "index": index, "elements": row["elements"]}
                        for index, row in enumerate((p for p in ports if p["kind"] == "flux"), 1)],
            "pickups": [{**row, "index": index} for index, row in enumerate(pickups, 1)],
        }
        design["sheets_wkt"].update({row["surface"]: planar[row["surface"]].wkt for row in pickups})
    return design, planar, nonplanar, volumes


def magnetostatic_config(design, groups, electrostatic):
    """Drive each bias route in turn and integrate signed B through passive SQUID apertures."""
    transfer = design["flux_transfer"]
    magnetostatic = {name: deepcopy(electrostatic[name]) for name in ("Model", "Domains", "Solver")}
    magnetostatic["Problem"] = {"Type": "Magnetostatic", "Output": "magnetostatic-output"}
    magnetostatic["Solver"]["Magnetostatic"] = {"InactivePorts": "Open"}
    magnetostatic["Boundaries"] = {
        "PEC": {"Attributes": [groups[name] for name in ("enclosure", *design["conductors"])]},
        "SurfaceCurrent": [
            {"Index": source["index"], "Elements": [
                {"Attributes": [groups[row["surface"]]], "Direction": row["direction"]}
                for row in source["elements"]]}
            for source in transfer["sources"]
        ],
        "Postprocessing": {"SurfaceFlux": [
            {"Index": row["index"], "Attributes": [groups[row["surface"]]], "Type": "Magnetic",
             "TwoSided": False, "Center": [*row["center_mm"], -.1]}
            for row in transfer["pickups"]
        ]},
    }
    return magnetostatic


def prepare_assembly(output, *, readout_topology="dedicated", refinement="coarse", frequency_grids=None,
                     chip=None, carrier=None, flux_transfer=False):
    """Freeze source layouts, generate the 3D assembly mesh, and write Palace inputs."""
    from .assembly_mesh import mesh_assembly
    from .carrier_layout import reference_carrier_design

    if refinement not in ("coarse", "fine") or readout_topology not in READOUT_TOPOLOGIES:
        raise ValueError("Choose a shared/dedicated layout and coarse/fine refinement")
    if (chip is None) != (carrier is None):
        raise ValueError("Supply both frozen die and carrier records or generate the reference pair")
    train, holdout = _frequency_grids(frequency_grids)
    if chip is None:
        dedicated = reference_hardware_design()
        shared = reference_hardware_design(readout_topology="shared")
        carrier = reference_carrier_design(dedicated, shared)
        chip = dedicated if readout_topology == "dedicated" else shared
    if chip["readout_topology"] != readout_topology:
        raise ValueError("The frozen die layout differs from the selected readout topology")
    design, planar, nonplanar, volumes = assembly_geometry(chip, carrier, flux_transfer=flux_transfer)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, record in zip(("design.json", *ASSEMBLY_SOURCES), (design, chip, carrier)):
        atomic_json(output / name, record)
    groups, materials, cells = mesh_assembly(output / "mesh.msh", planar=planar, nonplanar=nonplanar,
                                             refinement=refinement, **volumes)
    configs = _configs(design, groups, materials, 40, train, holdout)
    extra_outputs = None
    if flux_transfer:
        configs["magnetostatic"] = magnetostatic_config(design, groups, configs["electrostatic"])
        extra_outputs = {"magnetostatic": list(FLUX_TABLES)}
    return _write_field_inputs(output, design, configs, groups, materials, cells, refinement,
                               sources=ASSEMBLY_SOURCES, extra_outputs=extra_outputs)
