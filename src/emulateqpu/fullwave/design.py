"""Generate the four-data design and input files for a pinned Palace build."""
from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

from scipy.constants import c, elementary_charge, h, hbar

from ..chain import linear_chain_cell, linear_chain_geometry
from ..io import atomic_json, digest, file_hash

PALACE_TAG = "v0.18.0"
PALACE_COMMIT = "b92aef83ecfe6d360c4b3d83e2122986297f6778"
DESIGN_SCHEMA = "emulateqpu.fullwave.design.v2"
ASSEMBLY_DESIGN_SCHEMA = "emulateqpu.fullwave.assembly-design.v1"
LEGACY_DESIGN_SCHEMA = "emulateqpu.fullwave.design.v1"
DESIGN_ID = "four-data-line-four-readouts-v2"
SHARED_DESIGN_ID = "four-data-line-shared-readout-v2"
LEGACY_DESIGN_ID = "four-data-line-q1-readout-v1"
MODE_ORDER = ("q0", "c01", "q1", "c12", "q2", "c23", "q3")
READOUT_TOPOLOGIES = ("dedicated", "shared")
DATA_QUBITS = ("q0", "q1", "q2", "q3")
DEFAULT_READOUT_FREQUENCIES_GHZ = (6.45, 6.65, 6.85, 7.05)
DRIVE_PORTS = tuple(f"drive_{qubit}" for qubit in DATA_QUBITS)
PORT_ORDER = (*DRIVE_PORTS, *(f"readout_{qubit}_{end}"
                            for qubit in DATA_QUBITS for end in ("in", "out")))
LEGACY_PORT_ORDER = (*DRIVE_PORTS, "readout_in", "readout_out")
SHARED_PORT_ORDER = LEGACY_PORT_ORDER
ASSEMBLY_PORT_ORDER = (*PORT_ORDER, "flux_c01", "flux_c12", "flux_c23")
SUBSTRATE_PERMITTIVITY = 11.45
DRIVE_LAUNCH_Y_MM = -0.65
LADDER_MESHES = {
    "ladder-coarse": (0.018, 0.55),
    "ladder-medium": (0.012, 0.44),
    "ladder-fine": (0.008, 0.35),
}
LADDER_OUTPUT_DIRECTORY = "electrostatic-raw"


def ladder_mesh_settings(refinement, *, legacy=False):
    edge_size, volume_size = LADDER_MESHES[refinement]
    return {"conductor_edge_mm": edge_size, "bulk_mm": volume_size,
            "near_edge_mm": 0.006, "influence_mm": 0.090,
            "samples_per_curve": 500, "maximum_tetrahedra": 200_000 if legacy else 550_000}


def _chain_sheets(bounds, *, drive_gap_um=None):
    from shapely.geometry import box

    family = None
    if drive_gap_um is not None:
        family = {"ports": {str(i): {"drive": {"gap_um": drive_gap_um}} for i in range(0, len(MODE_ORDER), 2)}}
    design, base = linear_chain_geometry(4, family=family)
    assumed = linear_chain_cell(4)
    conductors = {}
    patches = {}
    junctions = []
    for index, label in enumerate(MODE_ORDER):
        left, right = design["terminal_mapping"][label]["islands"]
        conductors[left], conductors[right] = base[left], base[right]
        x = design["geometry"]["qubit_positions_um"][index][0] / 1000
        patches[f"jj_{label}"] = box(x - 0.015, -0.001, x + 0.015, 0.001)
        ej = assumed["EJ_idle_GHz"][index]
        junctions.append({"label": label, "port": 101 + index, "direction": "+X",
                          "EJ_GHz": ej, "bias_Phi0": assumed["idle_flux_Phi0"][index],
                          "inductance_H": (hbar / (2 * elementary_charge))**2 / (h * ej * 1e9),
                          "capacitance_F": 0.0})
    ports = []
    for index, label in enumerate(DRIVE_PORTS, start=1):
        pad = base[design["terminal_mapping"][label.removeprefix("drive_")]["drive"]]
        xlo, ylo, xhi, _ = pad.bounds
        if ylo <= DRIVE_LAUNCH_Y_MM:
            raise ValueError(f"Drive pickup for {label} extends past its launch")
        conductors[label] = pad.union(box(xlo, DRIVE_LAUNCH_Y_MM, xhi, ylo + 0.01))
        patches[label] = box(xlo, bounds[1], xhi, DRIVE_LAUNCH_Y_MM)
        ports.append({"label": label, "port": index, "direction": "+Y", "Z_ref_ohm": 50.0,
                      "reference_plane_mm": [0.5 * (xlo + xhi), bounds[1], 0.0]})
    return design, base, conductors, patches, junctions, ports


def _legacy_readout(bounds, design, base, conductors, patches, ports):
    from shapely.geometry import box

    pickup = base[design["terminal_mapping"]["q1"]["readout"]]
    xlo, ylo, xhi, _ = pickup.bounds
    conductors["readout_resonator"] = box(xlo, ylo, xhi, 9.4)
    conductors["readout_feedline"] = box(-3.45, 9.44, 3.45, 9.52)
    patches["readout_in"] = box(bounds[0], 9.44, -3.45, 9.52)
    patches["readout_out"] = box(3.45, 9.44, bounds[3], 9.52)
    ports += [
        {"label": "readout_in", "port": 5, "direction": "+X", "Z_ref_ohm": 50.0,
         "reference_plane_mm": [bounds[0], 9.48, 0.0]},
        {"label": "readout_out", "port": 6, "direction": "-X", "Z_ref_ohm": 50.0,
         "reference_plane_mm": [bounds[3], 9.48, 0.0]},
    ]


def _shared_ground_bridges(planes, wall_bonded, shared_path):
    from shapely.geometry import LineString

    bridges, bonded_by_bridge = [], set()
    for index, (start, stop) in enumerate(zip(shared_path[3:-1:4], shared_path[4::4])):
        x, y = (start[0] + stop[0]) / 2, start[1]
        width, half_span, deck_half_span, height = 0.024, 0.050, 0.024, 0.030
        profile = [[y - half_span, 0.0], [y - deck_half_span, height],
                   [y + deck_half_span, height], [y + half_span, 0.0]]
        feet = [LineString([(x - width / 2, foot_y), (x + width / 2, foot_y)])
                for foot_y in (profile[0][0], profile[-1][0])]
        islands = [[i for i, plane in enumerate(planes) if plane.covers(foot)] for foot in feet]
        if any(len(matches) != 1 for matches in islands):
            raise ValueError("Each shared-feedline ground bridge must land fully on ground")
        south, north = (matches[0] for matches in islands)
        if north not in wall_bonded or south in wall_bonded or south in bonded_by_bridge:
            raise ValueError("Each shared-feedline bridge must join a distinct interior ground to the enclosure ground")
        bonded_by_bridge.add(south)
        bridges.append({"label": f"ground_bridge_{index}", "center_mm": [x, y], "width_mm": width,
                        "profile_yz_mm": profile})
    if wall_bonded | bonded_by_bridge != set(range(len(planes))):
        raise ValueError("A shared-feedline ground segment is missing a ground bridge")
    return bridges


def _readout_sheets(bounds, design, base, conductors, patches, ports, *, shared, frequencies):
    from shapely import set_precision
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union

    width, cpw_gap = 0.010, 0.006
    coupling_gap = 0.010
    # Bring the U-bend close to the north wall to shorten its inner ground peninsula.
    coupling_start, coupling_end = 2.70, 3.10
    north = bounds[4]
    end_of_trace = north - 0.05
    join_y = north - 0.25
    effective_permittivity = (1 + SUBSTRATE_PERMITTIVITY) / 2
    clearances = {name: 0.040 for name in conductors}
    pickup_keepouts = []
    channels = []
    shared_path = []
    for qubit, frequency in zip(DATA_QUBITS, frequencies, strict=True):
        pickup = base[design["terminal_mapping"][qubit]["readout"]]
        pickup_x, pickup_y = pickup.centroid.x, pickup.bounds[1]
        left_leg, right_leg = pickup_x + 0.29, pickup_x + 0.68
        coupled_x = left_leg - width - coupling_gap
        has_input = not shared or qubit == DATA_QUBITS[0]
        has_output = not shared or qubit == DATA_QUBITS[-1]
        feedline_path = [(left_leg, end_of_trace if has_input else join_y),
                         (left_leg, coupling_start), (right_leg, coupling_start),
                         (right_leg, end_of_trace if has_output else join_y)]
        feedline_name = "readout_feedline" if shared else f"readout_{qubit}_feedline"
        resonator_name = f"readout_{qubit}_resonator"
        if shared:
            shared_path.extend(feedline_path)
        else:
            conductors[feedline_name] = LineString(feedline_path).buffer(width / 2, cap_style="flat", join_style="mitre")

        fanout_y, fold_bottom = 0.52, 1.00
        outer_fold_x, inner_fold_x = pickup_x - 0.47, pickup_x - 0.27

        def resonator_path(turn_y):
            return LineString([(pickup_x, pickup_y), (pickup_x, fanout_y),
                               (outer_fold_x, fanout_y), (outer_fold_x, turn_y),
                               (inner_fold_x, turn_y), (inner_fold_x, fold_bottom),
                               (coupled_x, fold_bottom), (coupled_x, coupling_end)])

        # The two long vertical runs each grow by the change in turn height.
        length_mm = c / (2 * frequency * 1e6 * math.sqrt(effective_permittivity))
        turn_y = coupling_end + (length_mm - resonator_path(coupling_end).length) / 2
        if not coupling_end < turn_y < end_of_trace:
            raise ValueError(f"Readout resonator for {qubit} does not fit its routing lane")
        centerline = resonator_path(turn_y)
        if not centerline.is_simple:
            raise ValueError(f"Readout resonator for {qubit} does not fit its routing lane")
        trace = centerline.buffer(width / 2, cap_style="flat", join_style="mitre")
        conductors[resonator_name] = pickup.union(trace)
        clearances[feedline_name] = clearances[resonator_name] = cpw_gap
        # Keep the qubit pickup exposed to the adjacent island.
        pickup_keepouts.append(pickup.buffer(0.040, join_style="mitre"))
        prefix = "readout" if shared else f"readout_{qubit}"
        input_port, output_port = f"{prefix}_in", f"{prefix}_out"
        for label, x, present in ((input_port, left_leg, has_input), (output_port, right_leg, has_output)):
            if not present:
                continue
            patches[label] = box(x - width / 2, end_of_trace, x + width / 2, north)
            ports.append({"label": label, "port": len(ports) + 1, "direction": "-Y", "Z_ref_ohm": 50.0,
                          "reference_plane_mm": [x, north, 0.0]})
        channels.append({"qubit": qubit, "input_port": input_port, "output_port": output_port,
                         "resonator": resonator_name, "feedline": feedline_name,
                         "centerline_length_mm": centerline.length, "nominal_frequency_GHz": frequency})
    if shared:
        path = LineString(shared_path)
        if not path.is_simple:
            raise ValueError("The shared feedline crosses itself")
        conductors["readout_feedline"] = path.buffer(width / 2, cap_style="flat", join_style="mitre")
    keepouts = [poly.buffer(clearances[name], join_style="mitre") for name, poly in conductors.items()]
    for name, patch in patches.items():
        clearance = cpw_gap if name.startswith("readout_") else 0.040
        keepouts.append(patch.buffer(clearance, join_style="mitre"))
    footprint = box(bounds[0], bounds[1], bounds[3], north)
    ground = set_precision(footprint.difference(unary_union(keepouts + pickup_keepouts)), 1e-6)
    planes = ground.geoms if ground.geom_type == "MultiPolygon" else (ground,)
    wall_bonded = {index for index, plane in enumerate(planes)
                   if plane.boundary.intersection(footprint.boundary).length >= 1e-6}
    bridges = _shared_ground_bridges(planes, wall_bonded, shared_path) if shared else []
    if not shared and len(wall_bonded) != len(planes):
        raise ValueError("A coplanar ground segment has no contact with the enclosure wall")
    conductors["chip_ground"] = ground
    return channels, {"trace_width_mm": width, "ground_gap_mm": cpw_gap,
                      "coupling_gap_mm": coupling_gap, "coupling_length_mm": coupling_end - coupling_start,
                      "length_estimate_effective_permittivity": effective_permittivity}, bridges


def _validate_readout_topology(legacy, readout_topology):
    if readout_topology not in READOUT_TOPOLOGIES:
        raise ValueError(f"Unknown readout topology {readout_topology!r}")
    if legacy and readout_topology == "shared":
        raise ValueError("The original single-readout design is incompatible with shared readout")


def add_planar_design_arguments(parser):
    parser.add_argument("--readout-frequencies-ghz", nargs=4, type=float, metavar=("Q0", "Q1", "Q2", "Q3"),
                        help="Nominal readout targets that set the four bare-die resonator lengths")
    parser.add_argument("--drive-gap-um", type=float, help="Common clearance between each XY pickup and its data island")


def _positive_layout_number(value, name):
    try:
        if type(value) in (int, float) and math.isfinite(value) and value > 0:
            return float(value)
    except OverflowError:
        pass
    raise ValueError(f"{name} must be a finite positive number")


def four_qubit_design(*, legacy=False, readout_topology="dedicated", readout_frequencies_GHz=None, drive_gap_um=None):
    """Return the ideal enclosure, conductor polygons, and oriented port patches in mm."""
    _validate_readout_topology(legacy, readout_topology)
    if legacy and (readout_frequencies_GHz is not None or drive_gap_um is not None):
        raise ValueError("Planar design controls require the four-resonator geometry")
    frequencies = DEFAULT_READOUT_FREQUENCIES_GHZ
    if readout_frequencies_GHz is not None:
        if not isinstance(readout_frequencies_GHz, (list, tuple)) or len(readout_frequencies_GHz) != len(DATA_QUBITS):
            raise ValueError("Provide four nominal readout frequencies in q0 through q3 order")
        frequencies = tuple(_positive_layout_number(value, f"Readout frequency for {qubit}")
                            for qubit, value in zip(DATA_QUBITS, readout_frequencies_GHz, strict=True))
    if drive_gap_um is not None:
        drive_gap_um = _positive_layout_number(drive_gap_um, "Drive gap")
        if drive_gap_um < 0.001:
            raise ValueError("Drive gap must be at least one 1 nm layout grid step")
        if drive_gap_um >= -1000 * DRIVE_LAUNCH_Y_MM:
            raise ValueError("Drive gap places the pickup beyond its launch")
    shared = readout_topology == "shared"
    bounds = ([-3.6, -0.8, -0.35, 3.6, 10.0, 0.7] if legacy
              else [-3.6, -0.8, -0.35, 4.1, 5.25, 0.7])
    design, base, conductors, patches, junctions, ports = _chain_sheets(bounds, drive_gap_um=drive_gap_um)
    channels = None
    if legacy:
        _legacy_readout(bounds, design, base, conductors, patches, ports)
    else:
        from shapely import set_precision

        channels, cpw, bridges = _readout_sheets(bounds, design, base, conductors, patches, ports, shared=shared,
                                                frequencies=frequencies)
        conductors = {name: set_precision(polygon, 1e-6) for name, polygon in conductors.items()}
        patches = {name: set_precision(polygon, 1e-6) for name, polygon in patches.items()}
    sheets = {**conductors, **{f"port_{key}": polygon for key, polygon in patches.items()}}
    names = list(sheets)
    contacts = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                if sheets[a].intersection(sheets[b]).area > 1e-10]
    if contacts or any(not poly.is_valid or poly.is_empty or
                       (name != "chip_ground" and (poly.geom_type != "Polygon" or poly.interiors))
                       for name, poly in sheets.items()):
        raise ValueError(f"Invalid planar sheets or overlapping copper and port patches: {contacts}")
    record = {"schema": LEGACY_DESIGN_SCHEMA if legacy else DESIGN_SCHEMA,
              "units": "mm", "mode_order": list(MODE_ORDER),
              "design_id": LEGACY_DESIGN_ID if legacy else DESIGN_ID, "enclosure_mm": bounds,
              "substrate_permittivity": SUBSTRATE_PERMITTIVITY, "junctions": junctions, "ports": ports,
              "island_order": [island for label in MODE_ORDER
                               for island in design["terminal_mapping"][label]["islands"]],
              "conductors": list(conductors), "sheets_wkt": {name: poly.wkt for name, poly in sheets.items()},
              "assumptions": {"metal": "perfect electric conductor", "substrate_loss_tangent": 0.0,
                              "readout": ("one q1-coupled open microstrip and a two-ended feedline" if legacy else
                                          "four open-ended half-wave CPW resonators on dedicated two-port feedlines"),
                              "junction_bias": "fixed idle flux, identical in field and electrostatic binding",
                              "port_type": "lumped 50-ohm sheet ports in a finite PEC enclosure"}}
    if channels is not None:
        record["readout_channels"] = channels
        record["readout_cpw"] = cpw
        record["assumptions"]["ground_return"] = "coplanar ground segments bonded at ideal PEC enclosure walls"
    if drive_gap_um is not None and drive_gap_um != design["geometry"]["drive_gap_um"]:
        record["drive_gap_um"] = drive_gap_um
    if shared:
        record["design_id"] = SHARED_DESIGN_ID
        record["readout_topology"] = "shared"
        record["ground_bridges"] = bridges
        record["assumptions"]["readout"] = "four open-ended half-wave CPW resonators on one two-port feedline"
        record["assumptions"]["ground_return"] = (
            "coplanar ground segments bonded by ideal enclosure walls and three PEC straps across the feedline"
        )
    return record, sheets


def _mesh_design(design, sheets, path, refinement):
    import gmsh

    legacy = design["schema"] == LEGACY_DESIGN_SCHEMA
    if refinement in LADDER_MESHES:
        settings = ladder_mesh_settings(refinement, legacy=legacy)
        edge_size, volume_size = settings["conductor_edge_mm"], settings["bulk_mm"]
        near_edge, influence = settings["near_edge_mm"], settings["influence_mm"]
        samples, maximum_tetrahedra = settings["samples_per_curve"], settings["maximum_tetrahedra"]
    else:
        scale = {"coarse": 1.0, "fine": 0.72}[refinement]
        sizes = (0.030, 0.65, 0.090) if legacy else (0.014, 0.42, 0.070)
        edge_size, volume_size, influence = (scale * size for size in sizes)
        near_edge, samples, maximum_tetrahedra = 0, 30, 300_000
    x0, y0, z0, x1, y1, z1 = design["enclosure_mm"]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add(design["design_id"])
        occ = gmsh.model.occ
        volumes = [(3, occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, -z0)),
                   (3, occ.addBox(x0, y0, 0, x1 - x0, y1 - y0, z1))]
        surfaces = []
        surface_indices = {}
        surface_areas = {}

        def curve_loop(coordinates):
            vertices = [occ.addPoint(x, y, z) for x, y, z in coordinates]
            lines = [occ.addLine(a, b) for a, b in zip(vertices, vertices[1:] + vertices[:1])]
            return occ.addCurveLoop(lines)

        for name, geometry in sheets.items():
            start = len(volumes) + len(surfaces)
            polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else (geometry,)
            for polygon in polygons:
                rings = [curve_loop((x, y, 0) for x, y in list(ring.coords)[:-1])
                         for ring in (polygon.exterior, *polygon.interiors)]
                surfaces.append((2, occ.addPlaneSurface(rings)))
            surface_indices[name] = slice(start, len(volumes) + len(surfaces))
            surface_areas[name] = geometry.area
        for bridge in design.get("ground_bridges", ()):
            name = bridge["label"]
            start = len(volumes) + len(surfaces)
            x, width = bridge["center_mm"][0], bridge["width_mm"]
            left, right = x - width / 2, x + width / 2
            profile = bridge["profile_yz_mm"]
            surface_areas[name] = 0.0
            for (ya, za), (yb, zb) in zip(profile, profile[1:]):
                loop = curve_loop([(left, ya, za), (right, ya, za), (right, yb, zb), (left, yb, zb)])
                surfaces.append((2, occ.addPlaneSurface([loop])))
                surface_areas[name] += width * math.hypot(yb - ya, zb - za)
            surface_indices[name] = slice(start, len(volumes) + len(surfaces))
        _, mapping = occ.fragment(volumes, surfaces)
        occ.synchronize()
        material_tags = {"substrate": 514, "air": 501}
        for (name, attribute), mapped in zip(material_tags.items(), mapping[:2]):
            gmsh.model.addPhysicalGroup(3, [tag for dim, tag in mapped if dim == 3], attribute, name)
        exterior = []
        for _, face in gmsh.model.getEntities(2):
            bounds = occ.getBoundingBox(2, face)
            planes = (x0, y0, z0, x1, y1, z1)
            if any(abs(bounds[i % 3] - plane) < 1e-6 and abs(bounds[i % 3 + 3] - plane) < 1e-6
                   for i, plane in enumerate(planes)):
                exterior.append(face)
        groups = {"enclosure": 1}
        gmsh.model.addPhysicalGroup(2, exterior, 1, "enclosure")
        used = set(exterior)
        edges = set()
        for index, (name, expected_area) in enumerate(surface_areas.items(), start=2):
            tags = sorted({tag for entries in mapping[surface_indices[name]] for dim, tag in entries if dim == 2})
            if not tags or used.intersection(tags):
                raise ValueError(f"Missing or ambiguous physical surface {name}")
            area = sum(occ.getMass(2, tag) for tag in tags)
            if abs(area - expected_area) > 1e-9:
                raise ValueError(f"Mesh surface area changed for {name}")
            used.update(tags)
            groups[name] = gmsh.model.addPhysicalGroup(2, tags, index, name)
            edges.update(tag for dim, tag in gmsh.model.getBoundary([(2, tag) for tag in tags], oriented=False)
                         if dim == 1)
        fields = gmsh.model.mesh.field
        fields.add("Distance", 1)
        fields.setNumbers(1, "CurvesList", sorted(edges))
        fields.setNumber(1, "Sampling", samples)
        fields.add("Threshold", 2)
        for key, value in {"InField": 1, "SizeMin": edge_size, "SizeMax": volume_size,
                           "DistMin": near_edge, "DistMax": influence}.items():
            fields.setNumber(2, key, value)
        fields.setAsBackgroundMesh(2)
        gmsh.option.setNumber("Mesh.MeshSizeMin", edge_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", volume_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(3)
        types, elements, _ = gmsh.model.mesh.getElements(3)
        count = sum(len(values) for values in elements)
        if set(types) != {4} or not 0 < count <= maximum_tetrahedra:
            raise ValueError(f"Expected up to {maximum_tetrahedra} linear tetrahedra, "
                             f"observed {count} with types {list(types)}")
        gmsh.write(str(path))
        return groups, material_tags, count
    finally:
        gmsh.finalize()


def _configs(design, groups, materials, n_modes, train_GHz, holdout_GHz):
    permittivity = design.get("material_permittivity")
    if permittivity is None:
        permittivity = {"substrate": design["substrate_permittivity"], "air": 1.0}
    common = {"Model": {"Mesh": "mesh.msh", "L0": .001},
              "Domains": {"Materials": [{"Attributes": [materials[name]], "Permittivity": eps}
                                        for name, eps in permittivity.items()]},
              "Solver": {"Order": 2, "Linear": {"Type": "Default", "Tol": 1e-10, "MaxIts": 400}}}
    ground = [groups["enclosure"], *(groups[name] for name in design["conductors"]),
              *(groups[bridge["label"]] for bridge in design.get("ground_bridges", ()))]
    static = deepcopy(common)
    static.update(Problem={"Type": "Electrostatic", "Output": "electrostatic-output"},
                  Boundaries={"Ground": {"Attributes": sorted(set(ground) - {groups[k] for k in design["island_order"]})},
                              "Terminal": [{"Index": i, "Attributes": [groups[k]]}
                                           for i, k in enumerate(design["island_order"], 1)]})
    jjs = [{"Index": row["port"], "Attributes": [groups[f"port_jj_{row['label']}"]],
            "Direction": row["direction"], "L": row["inductance_H"], "C": row["capacitance_F"]}
           for row in design["junctions"]]
    ports = []
    for row in design["ports"]:
        port = {"Index": row["port"], "R": row["Z_ref_ohm"]}
        if "elements" in row:
            port["Elements"] = [{"Attributes": [groups[element["surface"]]], "Direction": element["direction"]}
                                for element in row["elements"]]
        else:
            port.update(Attributes=[groups[f"port_{row['label']}"]], Direction=row["direction"])
        ports.append(port)
    eigen = deepcopy(common)
    eigen.update(Problem={"Type": "Eigenmode", "Output": "eigenmode-output",
                          "OutputFormats": {"GridFunction": True}},
                 Boundaries={"PEC": {"Attributes": ground}, "LumpedPort": ports + jjs})
    eigen["Solver"]["Eigenmode"] = {"N": n_modes, "Save": n_modes, "Target": 2.0, "Tol": 1e-9}
    configs = {"electrostatic": static, "eigenmode": eigen}
    for label, frequencies in [("driven-train", train_GHz), ("driven-holdout", holdout_GHz)]:
        driven = deepcopy(common)
        driven.update(Problem={"Type": "Driven", "Output": f"{label}-output"},
                      Boundaries={"PEC": {"Attributes": ground},
                                  "LumpedPort": [{**row, "Excitation": row["Index"]} for row in ports] + jjs})
        driven["Solver"]["Driven"] = {"Samples": [{"Type": "Point", "Freq": frequencies}]}
        configs[label] = driven
    return configs


def _electrostatic_options(config, refinement, order, full_assembly):
    config["Solver"]["Order"] = order
    if full_assembly:
        config["Solver"]["PartialAssemblyOrder"] = order + 1
    if refinement in LADDER_MESHES:
        config["Problem"]["Output"] = LADDER_OUTPUT_DIRECTORY


def _frequency_grids(frequency_grids):
    default_train = [round(1.8 + i * .1, 6) for i in range(83)]
    grids = frequency_grids if frequency_grids is not None else {
        "train_GHz": default_train, "holdout_GHz": [round(frequency + .05, 6) for frequency in default_train[:-1]]}
    if not isinstance(grids, dict) or set(grids) != {"train_GHz", "holdout_GHz"}:
        raise ValueError("Expected train_GHz and holdout_GHz grids")
    train, holdout = grids["train_GHz"], grids["holdout_GHz"]
    for grid in (train, holdout):
        if (not isinstance(grid, list) or len(grid) < 2
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in grid)
                or any(a >= b for a, b in zip(grid, grid[1:]))):
            raise ValueError("Grids require at least two strictly increasing positive frequencies")
    if set(train).intersection(holdout) or not train[0] < holdout[0] < holdout[-1] < train[-1]:
        raise ValueError("Held-out frequencies must be disjoint and strictly inside the training band")
    return train, holdout


def prepare_design(output, *, refinement="coarse", frequency_grids=None, electrostatic_order=2,
                   electrostatic_full_assembly=False, legacy=False, readout_topology="dedicated",
                   readout_frequencies_GHz=None, drive_gap_um=None):
    """Mesh a frozen geometry, write fresh Palace inputs, and record their SHA-256 hashes."""
    if refinement not in ("coarse", "fine", *LADDER_MESHES):
        raise ValueError(f"Unknown mesh refinement {refinement!r}")
    if type(electrostatic_order) is not int or electrostatic_order not in (1, 2):
        raise ValueError("Electrostatic order must be 1 or 2")
    train, holdout = _frequency_grids(frequency_grids)
    design, sheets = four_qubit_design(legacy=legacy, readout_topology=readout_topology,
                                      readout_frequencies_GHz=readout_frequencies_GHz, drive_gap_um=drive_gap_um)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "design.json", design)
    groups, materials, cells = _mesh_design(design, sheets, output / "mesh.msh", refinement)
    configs = _configs(design, groups, materials, 12 if legacy else 24, train, holdout)
    _electrostatic_options(configs["electrostatic"], refinement, electrostatic_order, electrostatic_full_assembly)
    settings = ladder_mesh_settings(refinement, legacy=legacy) if refinement in LADDER_MESHES else None
    return _write_field_inputs(output, design, configs, groups, materials, cells, refinement, settings=settings)


def _write_field_inputs(output, design, configs, groups, materials, cells, refinement, *, settings=None,
                        sources=(), extra_outputs=None):
    for name, config in configs.items():
        atomic_json(output / (name + ".json"), config)
    files = ["design.json", "mesh.msh", *(name + ".json" for name in configs), *sources]
    manifest = {"schema": "emulateqpu.fullwave.inputs.v1", "design_sha256": digest(design),
                "files_sha256": {name: file_hash(output / name) for name in files},
                "refinement": refinement, "tetrahedra": cells, "physical_surface_ids": groups, "material_ids": materials,
                "palace": {"tag": PALACE_TAG, "commit": PALACE_COMMIT},
                "expected_outputs": {"electrostatic": ["terminal-C.csv"],
                                     "eigenmode": ["eig.csv", "domain-E.csv", "port-EPR.csv", "port-Q.csv"],
                                     "driven-train": ["port-S.csv"], "driven-holdout": ["port-S.csv"]},
                "campaign_status": "MESH_ONLY"}
    if settings is not None:
        manifest["mesh_settings"] = settings
    if extra_outputs is not None:
        manifest["expected_outputs"].update(extra_outputs)
    atomic_json(output / "field-inputs.json", manifest)
    return manifest
