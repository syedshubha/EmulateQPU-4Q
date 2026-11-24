"""Conformal tetrahedral mesh of a recessed die, carrier, and package."""
from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

MATERIAL_IDS = {"air": 501, "silicon": 514, "alumina": 520, "attach": 521}


class _InvalidSignedQualities(ValueError):
    pass


def _intersection_volume(first, second):
    return math.prod(max(0, min(first[i + 3], second[i + 3]) - max(first[i], second[i])) for i in range(3))


def _surfaces(occ, planar, nonplanar):
    surfaces, slices, areas = [], {}, {}

    def loop(points):
        vertices = [occ.addPoint(*point) for point in points]
        curves = [occ.addLine(a, b) for a, b in zip(vertices, vertices[1:] + vertices[:1])]
        return occ.addCurveLoop(curves)

    for name, shape in planar.items():
        if shape.geom_type not in ("Polygon", "MultiPolygon") or not shape.is_valid or shape.is_empty:
            raise ValueError(f"Expected valid planar metal polygons for {name}")
        start = len(surfaces)
        polygons = shape.geoms if shape.geom_type == "MultiPolygon" else (shape,)
        for poly in polygons:
            rings = [loop((x, y, 0.0) for x, y in list(ring.coords)[:-1])
                     for ring in (poly.exterior, *poly.interiors)]
            outer, *holes = [(2, occ.addPlaneSurface([ring])) for ring in rings]
            # Cut explicitly so OCC's choice of wire orientation cannot fill a hole.
            faces = occ.cut([outer], holes)[0] if holes else [outer]
            surfaces.extend(faces)
        slices[name] = slice(start, len(surfaces))
        areas[name] = shape.area
    for name, quads in nonplanar.items():
        start = len(surfaces)
        for points in quads:
            if len(points) != 4 or any(len(point) != 3 or not all(map(math.isfinite, point)) for point in points):
                raise ValueError(f"Expected four finite XYZ vertices per ribbon face for {name}")
            surfaces.append((2, occ.addPlaneSurface([loop(points)])))
        slices[name] = slice(start, len(surfaces))
        areas[name] = sum(occ.getMass(2, tag) for _, tag in surfaces[slices[name]])
        if not areas[name] > 0:
            raise ValueError(f"Missing nonplanar ribbon area for {name}")
    return surfaces, slices, areas


def _mapped_tags(mapping, dim):
    return {tag for entries in mapping for dimension, tag in entries if dimension == dim}


def _check_measure(occ, dim, tags, expected, name):
    actual = sum(occ.getMass(dim, tag) for tag in tags)
    if not tags or not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError(f"Assembly fragment measure changed for {name}: expected {expected:g}, got {actual:g}")


def _check_tetrahedra(mesh):
    types, elements, _ = mesh.getElements(3)
    count = sum(map(len, elements))
    if set(types) != {4} or not 0 < count <= 850_000:
        raise ValueError(f"Expected up to 850000 linear tetrahedra, got {count} with types {list(types)}")
    quality = np.asarray(mesh.getElementQualities(elements[0], "minSICN"))
    if len(quality) != count:
        raise ValueError(f"Gmsh returned {len(quality)} signed qualities for {count} tetrahedra")
    invalid = np.count_nonzero(~np.isfinite(quality) | (quality <= 0))
    if invalid:
        raise _InvalidSignedQualities(f"Assembly mesh has {invalid} invalid signed qualities for {count} tetrahedra")
    return count


def mesh_assembly(path, *, bounds, die_bounds, carrier_bounds, pocket_bounds, attach_bounds,
                  planar, nonplanar, refinement):
    """Write a Gmsh mesh and return its named surface attributes, materials, and tetrahedron count."""
    import gmsh

    boxes = (bounds, die_bounds, carrier_bounds, pocket_bounds, attach_bounds)
    if any(len(box) != 6 or not all(map(math.isfinite, box)) or not all(box[i] < box[i + 3] for i in range(3))
           for box in boxes):
        raise ValueError("Assembly boxes require finite XYZ minima and maxima")
    if set(planar) & set(nonplanar) or "enclosure" in planar or "enclosure" in nonplanar:
        raise ValueError("Assembly surface names must be distinct from each other and the enclosure")
    if any(any(part[i] < bounds[i] or part[i + 3] > bounds[i + 3] for i in range(3))
           for part in (die_bounds, carrier_bounds, attach_bounds)):
        raise ValueError("The enclosure must contain the die, carrier, and attach layer")
    if _intersection_volume(die_bounds, attach_bounds) > 1e-9:
        raise ValueError("The silicon die and attach layer overlap")
    size_settings = {"coarse": (.014, .45, .070), "fine": (.010, .33, .055)}
    if refinement not in size_settings:
        raise ValueError(f"Unknown assembly mesh refinement {refinement!r}")
    edge_size, bulk_size, influence = size_settings[refinement]
    material_volume = {"silicon": _intersection_volume(die_bounds, die_bounds),
                       "alumina": _intersection_volume(carrier_bounds, carrier_bounds)
                                   - _intersection_volume(carrier_bounds, pocket_bounds),
                       "attach": _intersection_volume(attach_bounds, attach_bounds)}
    material_volume["air"] = _intersection_volume(bounds, bounds) - sum(material_volume.values())

    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("four-qubit-reference-assembly")
        occ = gmsh.model.occ

        def box_entity(box):
            return (3, occ.addBox(*box[:3], *(box[i + 3] - box[i] for i in range(3))))

        carrier, _ = occ.cut([box_entity(carrier_bounds)], [box_entity(pocket_bounds)])
        if not carrier:
            raise ValueError("The die pocket removed the entire carrier")
        volumes = [box_entity(bounds), box_entity(die_bounds), box_entity(attach_bounds), *carrier]
        surfaces, slices, areas = _surfaces(occ, planar, nonplanar)
        _, mapping = occ.fragment(volumes, surfaces)
        occ.synchronize()

        # The initial air block maps to both the air and the overlapping dielectric fragments.
        materials = {"silicon": _mapped_tags(mapping[1:2], 3), "attach": _mapped_tags(mapping[2:3], 3),
                     "alumina": _mapped_tags(mapping[3:len(volumes)], 3)}
        dielectric = set().union(*materials.values())
        if len(dielectric) != sum(map(len, materials.values())):
            raise ValueError("Assembly dielectric volumes overlap")
        materials["air"] = _mapped_tags(mapping[:1], 3) - dielectric
        if dielectric | materials["air"] != {tag for _, tag in gmsh.model.getEntities(3)}:
            raise ValueError("An assembly volume is missing its material")
        for name, tag in MATERIAL_IDS.items():
            _check_measure(occ, 3, materials[name], material_volume[name], name)
            gmsh.model.addPhysicalGroup(3, sorted(materials[name]), tag, name)

        model_volumes = gmsh.model.getEntities(3)
        domain_surfaces = {tag for dim, tag in gmsh.model.getBoundary(model_volumes, combined=False, oriented=False)
                           if dim == 2}
        domain_surfaces.update(tag for _, vol in model_volumes
                               for dim, tag in gmsh.model.mesh.getEmbedded(3, vol) if dim == 2)
        exterior = set()
        for face in domain_surfaces:
            face_bounds = occ.getBoundingBox(2, face)
            if any(abs(face_bounds[i % 3] - plane) < 1e-6 and abs(face_bounds[i % 3 + 3] - plane) < 1e-6
                   for i, plane in enumerate(bounds)):
                exterior.add(face)
        dx, dy, dz = (bounds[i + 3] - bounds[i] for i in range(3))
        _check_measure(occ, 2, exterior, 2 * (dx * dy + dx * dz + dy * dz), "enclosure")

        groups, used, edges = {"enclosure": 1}, set(), set()
        surface_map = mapping[len(volumes):]
        for index, (name, interval) in enumerate(slices.items(), start=2):
            tags = _mapped_tags(surface_map[interval], 2)
            if used & tags or tags - domain_surfaces:
                raise ValueError(f"Ambiguous or unbound assembly surface {name}")
            _check_measure(occ, 2, tags, areas[name], name)
            used.update(tags)
            faces = [(2, tag) for tag in sorted(tags)]
            edges.update(tag for dim, tag in gmsh.model.getBoundary(faces, combined=False, oriented=False)
                         if dim == 1)
            groups[name] = gmsh.model.addPhysicalGroup(2, sorted(tags), index, name)
        if not exterior - used:
            raise ValueError("The assembly needs a PEC enclosure surface")
        gmsh.model.addPhysicalGroup(2, sorted(exterior - used), 1, "enclosure")

        if edges:
            fields = gmsh.model.mesh.field
            distance = fields.add("Distance")
            fields.setNumbers(distance, "CurvesList", sorted(edges))
            fields.setNumber(distance, "Sampling", 24)
            threshold = fields.add("Threshold")
            for key, value in {"InField": distance, "SizeMin": edge_size, "SizeMax": bulk_size,
                               "DistMin": 0, "DistMax": influence}.items():
                fields.setNumber(threshold, key, value)
            fields.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber("Mesh.MeshSizeMin", edge_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", bulk_size)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.model.mesh.generate(3)
        try:
            count = _check_tetrahedra(gmsh.model.mesh)
        except _InvalidSignedQualities:
            # A second conformal algorithm can recover coplanar tets at the pocket rim.
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
            gmsh.model.mesh.generate(3)
            count = _check_tetrahedra(gmsh.model.mesh)
        destination = Path(path)
        with TemporaryDirectory(dir=destination.parent, prefix=".assembly-mesh-") as stage:
            candidate = Path(stage) / "mesh.msh"
            gmsh.write(str(candidate))
            gmsh.clear()
            gmsh.open(str(candidate))
            if _check_tetrahedra(gmsh.model.mesh) != count:
                raise ValueError("The serialized assembly mesh changed the tetrahedron count")
            candidate.replace(destination)
        return groups, MATERIAL_IDS.copy(), count
    finally:
        gmsh.finalize()
