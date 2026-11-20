"""Physical planar capacitive links for a user-configured transmon chip.

Each selected link is two separate connected metal arms, each tied to one
existing transmon island. Facing paddles remain separated by dielectric. No
coupler mode, airbridge, conductor crossing or microwave resonator is invented.
All physical coordinates and route waypoints are micrometres in the request.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from shapely import set_precision
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, box, mapping
from shapely.ops import substring, unary_union


ARCHITECTURE = 'fixed_direct_capacitive_links'
LINK_DEFAULTS = dict(lead_width_um=20., paddle_width_um=120., paddle_length_um=60., gap_um=30.)


def _positive(value, label):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be finite and positive')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be finite and positive') from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be finite and positive')
    return value


def numerical_profile(config):
    """Explicit, reproducible ground-buffer precision and 3D mesher choices."""
    grid = config.get('ground_precision_grid_mm', 0.)
    if isinstance(grid, bool) or not isinstance(grid, (int, float)) or not math.isfinite(grid) or grid not in (0., 1e-9):
        raise ValueError('ground_precision_grid_mm must be exactly 0 or 1e-9 mm')
    algorithm = config.get('mesh_algorithm_3d', 1)
    if isinstance(algorithm, bool) or not isinstance(algorithm, int) or algorithm not in (1, 10):
        raise ValueError('mesh_algorithm_3d must be integer 1 or 10')
    return float(grid), algorithm


def _pairs(raw, n):
    if not isinstance(raw, (list, tuple)):
        raise ValueError('capacitive_links.edges requires a list of distinct qubit pairs')
    pairs = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(isinstance(q, bool) or not isinstance(q, int) or not 0 <= q < n for q in pair):
            raise ValueError('Physical link endpoints must be in-range integer qubit pairs')
        pair = tuple(sorted(pair))
        if pair[0] == pair[1] or pair in pairs:
            raise ValueError('Physical capacitive links cannot be self edges or duplicates')
        pairs.append(pair)
    return sorted(pairs)


def normalized_links(config, layout):
    """Canonical physical link parameters, also used by geometry hashing."""
    raw = deepcopy(config.get('capacitive_links', {}))
    if not isinstance(raw, dict) or set(raw)-{'edges', 'routes', *LINK_DEFAULTS}:
        raise ValueError('Unsupported capacitive_links fields')
    n = layout['n_qubits']
    edges = _pairs(raw.get('edges', layout['intended_edges']), n)
    if edges != _pairs(layout['intended_edges'], n):
        raise ValueError('Physical capacitive_links.edges must exactly match intended_edges; update both when selecting connections')
    result = {name:_positive(raw.get(name, value), name) for name, value in LINK_DEFAULTS.items()}
    if result['lead_width_um'] > result['paddle_width_um']:
        raise ValueError('lead_width_um must not exceed paddle_width_um')
    routes = raw.get('routes', {})
    if not isinstance(routes, dict):
        raise ValueError('capacitive_links.routes must map canonical i-j edges to route objects')
    canonical_routes = {}
    allowed_keys = {f'{i}-{j}' for i, j in edges}
    for key, route in routes.items():
        if key not in allowed_keys or not isinstance(route, dict) or set(route)-{'islands', 'waypoints_um'}:
            raise ValueError('Explicit routes require selected canonical i-j keys and islands/waypoints_um fields')
        islands = route.get('islands')
        if islands is not None and (not isinstance(islands, list) or len(islands) != 2 or any(side not in ('a', 'b') for side in islands)):
            raise ValueError('Route islands must be two entries a or b in canonical qubit order')
        points = route.get('waypoints_um', [])
        if not isinstance(points, list) or len(points) > 32:
            raise ValueError('Route waypoints_um requires at most 32 intermediate [x,y] coordinates')
        values = np.asarray(points, float) if points else np.empty((0, 2))
        if values.shape != (len(points), 2) or not np.isfinite(values).all():
            raise ValueError('Route waypoints_um must contain finite [x,y] coordinates')
        canonical = dict(waypoints_um=values.tolist())
        if islands is not None:
            canonical['islands'] = islands
        canonical_routes[key] = canonical
    result.update(edges=[list(pair) for pair in edges], routes={key:canonical_routes[key] for key in sorted(canonical_routes)})
    return result


def default_chip9_geometry():
    """Best available tested finite design for engineering demonstrations.

    The P2 fine extraction passes G0, but same-order spatial sensitivity remains
    material. This factory is not a numerical-accuracy or hardware qualification.
    """
    return normalize_chip9_geometry(dict(architecture=ARCHITECTURE, n_qubits=9, topology="grid",
        ground_clearance_um=41., ground_precision_grid_mm=1e-9, mesh_algorithm_3d=10,
        mesh_near_um=40., mesh_far_um=300., solver_order=2,
        capacitive_links=dict(lead_width_um=10., paddle_width_um=40., paddle_length_um=40., gap_um=150.)))


def normalize_chip9_geometry(geometry):
    """Resolve a complete physical design; this does not mesh or solve it."""
    from .layout import make_layout_config, resolve_layout
    if not isinstance(geometry, dict):
        raise ValueError('Physical geometry must be an object')
    if not geometry:
        return default_chip9_geometry()
    supplied = deepcopy(geometry)
    numerical_profile(supplied)
    if supplied.get('architecture', ARCHITECTURE) != ARCHITECTURE:
        raise ValueError('This physical graph workflow requires fixed_direct_capacitive_links architecture')
    defaults = dict(architecture=ARCHITECTURE, n_qubits=9, topology='grid',
                    mesh_near_um=60., mesh_far_um=450., solver_order=1,
                    mesh_timeout_s=300, solver_timeout_s=900, max_rss_gib=3., threads=1,
                    ground_policy='boundary_connected_remove_enclosed_islands')
    defaults.update(supplied)
    if defaults.get('ground_policy') != 'boundary_connected_remove_enclosed_islands':
        raise ValueError('Physical capacitive-link mode requires its explicit boundary-connected ground policy')
    cfg = make_layout_config(config=defaults)
    layout = resolve_layout(cfg)
    cfg['intended_edges'] = deepcopy(layout['intended_edges'])
    cfg['capacitive_links'] = normalized_links(cfg, layout)
    # Complete the signature-visible link defaults before a second automatic
    # footprint calculation, including user route waypoints.
    return make_layout_config(config=cfg)


def _island_center(config, layout, q, side):
    angle = math.radians(layout['orientations_deg'][q])
    offset = (float(config['pad_width_um'])+float(config['junction_gap_um']))/2
    offset *= -1 if side == 'a' else 1
    x, y = layout['positions_um'][q]
    return [x+math.cos(angle)*offset, y+math.sin(angle)*offset]


def link_centerlines(config, layout):
    """Resolve endpoint islands and full centreline routes without producing CAD."""
    links = normalized_links(config, layout)
    result = []
    for i, j in links['edges']:
        a, b = np.asarray(layout['positions_um'][i]), np.asarray(layout['positions_um'][j])
        dx, dy = b-a
        route = links['routes'].get(f'{i}-{j}', {})
        aligned_vertical = abs(dx) < 1e-9 and abs(dy) > 1e-9 and all(abs(layout['orientations_deg'][q] % 360) < 1e-9 for q in (i, j))
        if aligned_vertical:
            # The north arm leaves island a and the south arm island b,
            # avoiding each cell's drive-below-a and readout-above-b interfaces.
            sides = ['a', 'b'] if dy > 0 else ['b', 'a']
        else:
            sides = []
            for q, direction in ((i, b-a), (j, a-b)):
                theta = math.radians(layout['orientations_deg'][q])
                sides.append('b' if np.dot(direction, [math.cos(theta), math.sin(theta)]) >= 0 else 'a')
        sides = route.get('islands', sides)
        start, end = _island_center(config, layout, i, sides[0]), _island_center(config, layout, j, sides[1])
        if 'waypoints_um' in route:
            waypoints = route['waypoints_um']
        elif aligned_vertical:
            middle_y = (a[1]+b[1])/2
            sign = 1 if dy > 0 else -1
            reach = min(abs(dy)/6, 200.)
            middle_x = (start[0]+end[0])/2
            waypoints = [[start[0], middle_y-sign*reach], [middle_x, middle_y-sign*reach],
                         [middle_x, middle_y+sign*reach], [end[0], middle_y+sign*reach]]
        else:
            waypoints = []
        points = [start]+waypoints+[end]
        if any(math.dist(left, right) <= 1e-9 for left, right in zip(points[:-1], points[1:])):
            raise ValueError(f'Physical link {i}-{j} contains a zero-length route segment')
        line = LineString(points)
        if not line.is_simple:
            raise ValueError(f'Physical link {i}-{j} route crosses itself; adjust waypoints or placement')
        result.append(dict(qubits=[i, j], islands=sides, points_um=points, line=line))
    return result


def route_footprint_points(config, layout):
    links = normalized_links(config, layout)
    margin = math.hypot(links['paddle_width_um'], links['paddle_length_um'])/2+links['lead_width_um']
    return [(x+sx*margin, y+sy*margin) for route in link_centerlines(config, layout)
            for x, y in route['points_um'] for sx in (-1, 1) for sy in (-1, 1)]


def _arms(config, layout):
    links = normalized_links(config, layout)
    arms, records = {}, []
    for route in link_centerlines(config, layout):
        line = route['line']
        i, j = route['qubits']
        middle = line.length/2
        half_gap = links['gap_um']/2
        length = links['paddle_length_um']
        if middle <= half_gap+length+links['lead_width_um']:
            raise ValueError(f'Physical link {i}-{j} is too short for the requested gap and paddles')
        lo, hi = middle-half_gap-length, middle+half_gap+length
        central = substring(line, lo, hi)
        begin, end = np.array(central.coords[0]), np.array(central.coords[-1])
        if abs(np.linalg.norm(end-begin)-central.length) > 1e-7:
            raise ValueError(f'Physical link {i}-{j} needs a straight central segment across the coupling gap/paddles; adjust route waypoints')
        angle = math.degrees(math.atan2(*(end-begin)[::-1]))
        pair_paddles = []
        for endpoint, distance, side in [(0, middle-half_gap-length/2, route['islands'][0]),
                                         (1, middle+half_gap+length/2, route['islands'][1])]:
            centre = line.interpolate(distance)
            paddle = translate(rotate(box(-length/2, -links['paddle_width_um']/2,
                                          length/2, links['paddle_width_um']/2), angle, origin=(0, 0)), centre.x, centre.y)
            lead_line = substring(line, 0, distance) if endpoint == 0 else substring(line, distance, line.length)
            lead = lead_line.buffer(links['lead_width_um']/2, cap_style=2, join_style=2)
            arm = unary_union([lead, paddle])
            if arm.geom_type != 'Polygon' or not arm.is_valid:
                raise ValueError(f'Physical link {i}-{j} has a disconnected or invalid metal arm')
            label = f'q{route["qubits"][endpoint]}_{side}'
            name = f'link_{i}_{j}_{endpoint}'
            # Actual Qiskit Metal lengths use millimetres.
            from shapely.affinity import scale
            arms[name] = dict(label=label, geometry=scale(arm, xfact=.001, yfact=.001, origin=(0, 0)))
            pair_paddles.append(paddle)
        actual_gap = pair_paddles[0].distance(pair_paddles[1])
        if abs(actual_gap-links['gap_um']) > 1e-7:
            raise ValueError(f'Physical link {i}-{j} does not realize the requested dielectric gap')
        records.append(dict(qubits=[i, j], islands=route['islands'], centreline_um=route['points_um'],
                            actual_paddle_gap_um=actual_gap, gap_is_conductor=False,
                            quantum_coupler_mode=False))
    return arms, records


def connected_ground(config, terminals):
    """Keep boundary-connected ground and explicitly remove floating interiors."""
    if config.get('ground_policy', 'boundary_connected_remove_enclosed_islands') != 'boundary_connected_remove_enclosed_islands':
        raise ValueError('Unsupported physical capacitive-link ground policy')
    chip = box(-float(config['chip_width_um'])/2000, -float(config['chip_height_um'])/2000,
                float(config['chip_width_um'])/2000, float(config['chip_height_um'])/2000)
    clearance = float(config['ground_clearance_um'])/1000
    grid, algorithm = numerical_profile(config)
    # Only the buffered clearance cutout is normalized. The terminal polygons
    # are read-only and retain their original exact WKB/physical identities.
    nominal_cutout = unary_union(list(terminals.values())).buffer(clearance, join_style=2)
    normalized_cutout = set_precision(nominal_cutout, grid) if grid else nominal_cutout
    symmetric_difference = nominal_cutout.symmetric_difference(normalized_cutout).area
    relative_area = symmetric_difference/nominal_cutout.area
    hausdorff_um = nominal_cutout.hausdorff_distance(normalized_cutout)*1000
    if relative_area > 1e-9 or hausdorff_um > 2e-6:
        raise ValueError('Ground precision normalization exceeds frozen area/clearance perturbation tolerances')
    def boundary_connected(cutout):
        raw = chip.difference(cutout)
        components = [raw] if raw.geom_type == 'Polygon' else list(raw.geoms)
        boundary = [polygon for polygon in components if polygon.boundary.intersects(chip.boundary)]
        if len(boundary) != 1:
            raise ValueError('Physical routing disconnects the boundary ground plane; enlarge chip or change routes')
        ground = boundary[0]
        removed = [polygon for polygon in components if not polygon.equals(ground)]
        return ground, removed
    ground, removed = boundary_connected(normalized_cutout)
    nominal_ground, nominal_removed = boundary_connected(nominal_cutout) if grid else (ground, removed)
    ground_difference = nominal_ground.symmetric_difference(ground).area
    relative_ground_difference = ground_difference/nominal_ground.area
    if relative_ground_difference > 1e-9:
        raise ValueError('Ground precision changes boundary-connected ground topology beyond the frozen whole-ground relative area tolerance 1e-9 '
                         f'({ground_difference:.12g} mm2, relative {relative_ground_difference:.12g}; '
                         f'enclosed components {len(nominal_removed)} -> {len(removed)}). '
                         'Revise the physical clearance/routing away from degenerate point contacts; no mesh was executed.')
    clearance_changes = {label:(polygon.distance(ground)-polygon.distance(nominal_ground))*1000
                         for label, polygon in terminals.items()}
    max_clearance_change = max(abs(value) for value in clearance_changes.values())
    if max_clearance_change > 2e-6:
        raise ValueError('Ground precision normalization exceeds frozen per-terminal clearance-change tolerance')
    precision = dict(grid_mm=grid, mesh_algorithm_3d=algorithm,
        applied_to='Buffered ground-clearance cutout only; original terminal/island polygons unchanged',
        nominal_cutout_area_mm2=nominal_cutout.area, normalized_cutout_area_mm2=normalized_cutout.area,
        symmetric_difference_area_mm2=symmetric_difference, symmetric_difference_relative_area=relative_area,
        hausdorff_distance_um=hausdorff_um, per_terminal_clearance_change_um=clearance_changes,
        maximum_absolute_clearance_change_um=max_clearance_change,
        nominal_ground_area_mm2=nominal_ground.area, normalized_ground_area_mm2=ground.area,
        nominal_removed_enclosed_islands=len(nominal_removed), normalized_removed_enclosed_islands=len(removed),
        ground_symmetric_difference_area_mm2=ground_difference,
        ground_symmetric_difference_relative_area=relative_ground_difference,
        gates=dict(relative_cutout_area_tolerance=1e-9, relative_whole_ground_area_tolerance=1e-9,
                   clearance_tolerance_um=2e-6, passed=True))
    return ground, dict(policy='boundary_connected_remove_enclosed_islands', precision_normalization=precision,
        removed_enclosed_islands=len(removed), removed_area_um2=sum(polygon.area for polygon in removed)*1e6,
        removed_polygons_mm=[mapping(polygon) for polygon in removed],
        scope='Enclosed floating metal is deliberately absent from this planar design; no airbridges, ground bonds or extra grounded terminals are assumed')
