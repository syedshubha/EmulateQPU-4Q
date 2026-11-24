"""Expose the assembled SQUID openings as passive magnetic flux surfaces."""
from __future__ import annotations


def add_flux_apertures(chip, planar):
    """Insert three witness polygons at z=0 and return their metadata.

    Close each loop using its two field-model islands and two lumped junction
    sheets, then use the enclosed hole as the integration surface. Its boundary
    follows the inside edges of the homogenized circuit. Coordinates and areas
    are in mm and mm². These sheets only identify integration surfaces and must
    receive neither a PEC boundary condition nor a drive port.
    """
    from shapely.geometry import Point, Polygon
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union

    loops = chip["squid_loops"]
    if len(loops) != 3 or len({row["coupler"] for row in loops}) != 3:
        raise ValueError("Expected three distinct SQUID loops for flux integration")
    devices = {row["label"]: row for row in chip["junction_devices"]}
    apertures, records = {}, []
    for loop in loops:
        coupler = loop["coupler"]
        labels = list(loop["junctions"])
        branches = [devices.get(label) for label in labels]
        if (len(set(labels)) != 2 or len(labels) != 2
                or any(branch is None or branch["mode"] != coupler for branch in branches)):
            raise ValueError(f"Expected two junction branches for SQUID {coupler}")
        islands = list(branches[0]["islands"])
        if len(set(islands)) != 2 or len(islands) != 2 or islands != list(branches[1]["islands"]):
            raise ValueError(f"SQUID {coupler} junctions must connect the same two islands")
        junction_surfaces = [f"port_jj_{label}" for label in labels]
        boundary_names = {*islands, *junction_surfaces}
        if missing := boundary_names - planar.keys():
            raise ValueError(f"Missing assembled sheets for SQUID {coupler}: {sorted(missing)}")
        closed = unary_union([planar[name] for name in (*islands, *junction_surfaces)])
        if closed.geom_type != "Polygon" or not closed.is_valid or len(closed.interiors) != 1:
            raise ValueError(f"SQUID {coupler} must enclose exactly one assembled aperture")
        aperture = orient(Polygon(closed.interiors[0]), sign=1.0)
        if not aperture.is_valid or not aperture.contains(Point(*loop["center_mm"])):
            raise ValueError(f"SQUID {coupler} aperture must contain the layout loop center")

        contacts = {name: aperture.intersection(sheet) for name, sheet in planar.items()
                    if aperture.intersects(sheet)}
        if (set(contacts) != boundary_names
                or any(edge.area > 0 or edge.length <= 0 for edge in contacts.values())
                or not unary_union(list(contacts.values())).equals(aperture.boundary)):
            raise ValueError(f"SQUID {coupler} aperture must touch only its exact island and junction edges")
        surface = f"flux_pickup_{coupler}"
        if surface in planar or any(aperture.intersects(other) for other in apertures.values()):
            raise ValueError(f"SQUID {coupler} flux surface conflicts with another sheet")
        apertures[surface] = aperture
        records.append({"label": coupler, "coupler": coupler, "surface": surface, "direction": "+Z",
                        "area_mm2": aperture.area, "center_mm": list(aperture.centroid.coords[0]),
                        "junctions": labels, "junction_surfaces": junction_surfaces, "islands": islands})

    planar.update(apertures)
    return records
