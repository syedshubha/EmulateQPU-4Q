"""Generate the four-data design and input files for a pinned Palace build."""
from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

from scipy.constants import elementary_charge, h, hbar

from ..chain import linear_chain_cell, linear_chain_geometry
from ..io import atomic_json, digest, file_hash

PALACE_TAG = "v0.18.0"
PALACE_COMMIT = "b92aef83ecfe6d360c4b3d83e2122986297f6778"
DESIGN_SCHEMA = "emulateqpu.fullwave.design.v1"
MODE_ORDER = ("q0", "c01", "q1", "c12", "q2", "c23", "q3")
PORT_ORDER = ("drive_q0", "drive_q1", "drive_q2", "drive_q3", "readout_in", "readout_out")
LADDER_MESHES = {
    "ladder-coarse": (0.018, 0.55),
    "ladder-medium": (0.012, 0.44),
    "ladder-fine": (0.008, 0.35),
}
LADDER_OUTPUT_DIRECTORY = "electrostatic-raw"


def ladder_mesh_settings(refinement):
    edge_size, volume_size = LADDER_MESHES[refinement]
    return {"conductor_edge_mm": edge_size, "bulk_mm": volume_size,
            "near_edge_mm": 0.006, "influence_mm": 0.090,
            "samples_per_curve": 500, "maximum_tetrahedra": 200_000}


def four_qubit_design():
    """Return the ideal enclosure, conductor polygons, and oriented port patches in mm."""
    from shapely.geometry import box

    design, base = linear_chain_geometry(4)
    assumed = linear_chain_cell(4)
    bounds = [-3.6, -0.8, -0.35, 3.6, 10.0, 0.7]
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
    for index, label in enumerate(PORT_ORDER[:4], start=1):
        pad = base[design["terminal_mapping"][label.removeprefix("drive_")]["drive"]]
        xlo, ylo, xhi, _ = pad.bounds
        conductors[label] = pad.union(box(xlo, -0.65, xhi, ylo + 0.01))
        patches[label] = box(xlo, bounds[1], xhi, -0.65)
        ports.append({"label": label, "port": index, "direction": "+Y", "Z_ref_ohm": 50.0,
                      "reference_plane_mm": [0.5 * (xlo + xhi), bounds[1], 0.0]})
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
    sheets = {**conductors, **{f"port_{key}": polygon for key, polygon in patches.items()}}
    names = list(sheets)
    contacts = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                if sheets[a].intersection(sheets[b]).area > 1e-10]
    if contacts or any(poly.geom_type != "Polygon" or not poly.is_valid or poly.interiors
                       for poly in sheets.values()):
        raise ValueError(f"Invalid planar sheets or overlapping copper and port patches: {contacts}")
    record = {"schema": DESIGN_SCHEMA, "units": "mm", "mode_order": list(MODE_ORDER),
              "design_id": "four-data-line-q1-readout-v1", "enclosure_mm": bounds,
              "substrate_permittivity": 11.45, "junctions": junctions, "ports": ports,
              "island_order": [island for label in MODE_ORDER
                               for island in design["terminal_mapping"][label]["islands"]],
              "conductors": list(conductors), "sheets_wkt": {name: poly.wkt for name, poly in sheets.items()},
              "assumptions": {"metal": "perfect electric conductor", "substrate_loss_tangent": 0.0,
                              "readout": "one q1-coupled open microstrip and a two-ended feedline",
                              "junction_bias": "fixed idle flux, identical in field and electrostatic binding",
                              "port_type": "lumped 50-ohm sheet ports in a finite PEC enclosure"}}
    return record, sheets


def _mesh_design(design, sheets, path, refinement):
    import gmsh

    if refinement in LADDER_MESHES:
        settings = ladder_mesh_settings(refinement)
        edge_size, volume_size = settings["conductor_edge_mm"], settings["bulk_mm"]
        near_edge, influence = settings["near_edge_mm"], settings["influence_mm"]
        samples, maximum_tetrahedra = settings["samples_per_curve"], settings["maximum_tetrahedra"]
    else:
        scale = {"coarse": 1.0, "fine": 0.72}[refinement]
        edge_size, volume_size, influence = (scale * size for size in (0.030, 0.65, 0.090))
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
        for polygon in sheets.values():
            vertices = [occ.addPoint(x, y, 0) for x, y in list(polygon.exterior.coords)[:-1]]
            lines = [occ.addLine(start, stop) for start, stop in zip(vertices, vertices[1:] + vertices[:1])]
            surfaces.append((2, occ.addPlaneSurface([occ.addCurveLoop(lines)])))
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
        for index, (name, polygon) in enumerate(sheets.items(), start=2):
            tags = [tag for dim, tag in mapping[index] if dim == 2]
            if not tags or used.intersection(tags):
                raise ValueError(f"Missing or ambiguous physical surface {name}")
            area = sum(occ.getMass(2, tag) for tag in tags)
            if abs(area - polygon.area) > 1e-9:
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
    common = {"Model": {"Mesh": "mesh.msh", "L0": .001},
              "Domains": {"Materials": [
                  {"Attributes": [materials["substrate"]], "Permittivity": design["substrate_permittivity"]},
                  {"Attributes": [materials["air"]], "Permittivity": 1.0}]},
              "Solver": {"Order": 2, "Linear": {"Type": "Default", "Tol": 1e-10, "MaxIts": 400}}}
    ground = [groups["enclosure"], *(groups[name] for name in design["conductors"])]
    static = deepcopy(common)
    static.update(Problem={"Type": "Electrostatic", "Output": "electrostatic-output"},
                  Boundaries={"Ground": {"Attributes": sorted(set(ground) - {groups[k] for k in design["island_order"]})},
                              "Terminal": [{"Index": i, "Attributes": [groups[k]]}
                                           for i, k in enumerate(design["island_order"], 1)]})
    jjs = [{"Index": row["port"], "Attributes": [groups[f"port_jj_{row['label']}"]],
            "Direction": row["direction"], "L": row["inductance_H"], "C": row["capacitance_F"]}
           for row in design["junctions"]]
    ports = [{"Index": row["port"], "Attributes": [groups[f"port_{row['label']}"]],
              "Direction": row["direction"], "R": row["Z_ref_ohm"]} for row in design["ports"]]
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


def prepare_design(output, *, refinement="coarse", frequency_grids=None, electrostatic_order=2,
                   electrostatic_full_assembly=False):
    """Mesh a frozen geometry, write fresh Palace inputs, and record their SHA-256 hashes."""
    if refinement not in ("coarse", "fine", *LADDER_MESHES):
        raise ValueError(f"Unknown mesh refinement {refinement!r}")
    if type(electrostatic_order) is not int or electrostatic_order not in (1, 2):
        raise ValueError("Electrostatic order must be 1 or 2")
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
    design, sheets = four_qubit_design()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "design.json", design)
    groups, materials, cells = _mesh_design(design, sheets, output / "mesh.msh", refinement)
    configs = _configs(design, groups, materials, 12, train, holdout)
    _electrostatic_options(configs["electrostatic"], refinement, electrostatic_order, electrostatic_full_assembly)
    for name, config in configs.items():
        atomic_json(output / (name + ".json"), config)
    files = ["design.json", "mesh.msh", *(name + ".json" for name in configs)]
    manifest = {"schema": "emulateqpu.fullwave.inputs.v1", "design_sha256": digest(design),
                "files_sha256": {name: file_hash(output / name) for name in files},
                "refinement": refinement, "tetrahedra": cells, "physical_surface_ids": groups, "material_ids": materials,
                "palace": {"tag": PALACE_TAG, "commit": PALACE_COMMIT},
                "expected_outputs": {"electrostatic": ["terminal-C.csv"],
                                     "eigenmode": ["eig.csv", "domain-E.csv", "port-EPR.csv", "port-Q.csv"],
                                     "driven-train": ["port-S.csv"], "driven-holdout": ["port-S.csv"]},
                "campaign_status": "MESH_ONLY"}
    if refinement in LADDER_MESHES:
        manifest["mesh_settings"] = ladder_mesh_settings(refinement)
    atomic_json(output / "field-inputs.json", manifest)
    return manifest
