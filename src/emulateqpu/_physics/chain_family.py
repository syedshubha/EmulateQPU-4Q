"""Parametric chain conductor geometry and island capacitance reduction."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np
from scipy.constants import e, h
from shapely.affinity import rotate, scale, translate
from shapely.geometry import box
from shapely.ops import substring, unary_union

from .chip9_geometry import (ARCHITECTURE, link_centerlines, normalize_chip9_geometry)
from .layout import resolve_layout

FAMILY_SCHEMA = 'emulateqpu.chain_family.v1'
LINK_KEYS = ('lead_width_um', 'paddle_width_um', 'paddle_length_um', 'gap_um')
PORT_KINDS = ('drive', 'readout')
PORT_KEYS = ('pad', 'side', 'gap_um', 'offset_um', 'width_um', 'length_um')
MESH_PROFILES = {
    'p2_e12': dict(mesh_edge_um=12., mesh_near_um=45., mesh_far_um=250., mesh_edge_transition_um=60.),
    'p2_e8': dict(mesh_edge_um=8., mesh_near_um=35., mesh_far_um=220., mesh_edge_transition_um=60.),
}


def mode_order(n_data):
    if n_data not in (3, 5): raise ValueError('Chain family supports 3 or 5 data transmons')
    out = []
    for k in range(n_data):
        out.append(f'q{k}')
        if k < n_data-1: out.append(f'c{k}{k+1}')
    return out


def _finite_positive(value, label, allow_zero=False):
    if isinstance(value, bool): raise ValueError(label+' must be a number')
    v = float(value)
    if not math.isfinite(v) or (v < 0 if allow_zero else v <= 0): raise ValueError(label+' is outside its physical range')
    return v


def normalize_family(family, n_pads):
    """Canonical family overrides; unknown keys, pads or edges are rejected."""
    f = deepcopy(family or {})
    if set(f)-{'schema', 'edge_links', 'ports', 'name', 'notes'}: raise ValueError('Unsupported chain-family fields')
    if f.get('schema', FAMILY_SCHEMA) != FAMILY_SCHEMA: raise ValueError('Unsupported chain-family schema')
    edges = {f'{i}-{i+1}' for i in range(n_pads-1)}
    links = {}
    for key, value in sorted((f.get('edge_links') or {}).items()):
        if key not in edges or not isinstance(value, dict) or not value or set(value)-set(LINK_KEYS):
            raise ValueError(f'edge_links[{key}] must override link sizes of an existing neighbour edge')
        links[key] = {k: _finite_positive(v, f'edge_links[{key}].{k}') for k, v in sorted(value.items())}
    ports = {}
    for pad, spec in sorted((f.get('ports') or {}).items(), key=lambda kv: int(kv[0])):
        p = int(pad)
        if str(p) != str(pad) or not 0 <= p < n_pads or not isinstance(spec, dict) or not spec or set(spec)-set(PORT_KINDS):
            raise ValueError(f'ports[{pad}] must name an existing pad and drive/readout overrides')
        entry = {}
        for kind, o in sorted(spec.items()):
            if not isinstance(o, dict) or not o or set(o)-set(PORT_KEYS): raise ValueError(f'ports[{pad}].{kind} has unsupported fields')
            c = {}
            for k, v in sorted(o.items()):
                if k == 'pad':
                    if v not in ('a', 'b'): raise ValueError('port pad must be a or b')
                    c[k] = v
                elif k == 'side':
                    if v not in ('above', 'below'): raise ValueError('port side must be above or below')
                    c[k] = v
                elif k == 'offset_um':
                    if isinstance(v, bool) or not math.isfinite(float(v)): raise ValueError('offset_um must be finite')
                    c[k] = float(v)
                else:
                    c[k] = _finite_positive(v, f'ports[{pad}].{kind}.{k}')
            entry[kind] = c
        ports[str(p)] = entry
    return dict(schema=FAMILY_SCHEMA, edge_links=links, ports=ports)


def family_sha256(family):
    return hashlib.sha256(json.dumps(family, sort_keys=True, allow_nan=False).encode()).hexdigest()


def edge_link_params(config, family, i, j):
    base = {k: float(config['capacitive_links'][k]) for k in LINK_KEYS}
    base.update(family['edge_links'].get(f'{min(i, j)}-{max(i, j)}', {}))
    if base['lead_width_um'] > base['paddle_width_um']: raise ValueError(f'Edge {i}-{j}: lead wider than paddle')
    return base


def port_local(config, family, pad, kind):
    """(x, y, width, length) of a port rectangle centre in the pad's local frame, micrometres.

    Defaults equal em_worker.chip_cad: drive below island a, readout above island b, centred on that island.
    """
    w, h, jg = (float(config[k]) for k in ('pad_width_um', 'pad_height_um', 'junction_gap_um'))
    d = dict(pad='a' if kind == 'drive' else 'b', side='below' if kind == 'drive' else 'above',
             gap_um=float(config[kind+'_gap_um']), offset_um=0., width_um=float(config[kind+'_width_um']),
             length_um=float(config[kind+'_length_um']))
    d.update(family['ports'].get(str(pad), {}).get(kind, {}))
    x = (-1 if d['pad'] == 'a' else 1)*(w+jg)/2+d['offset_um']
    y = (h/2+d['gap_um']+d['length_um']/2)*(-1 if d['side'] == 'below' else 1)
    return x, y, d['width_um'], d['length_um'], d


def _to_global(layout, pad, x, y):
    px, py = layout['positions_um'][pad]; a = math.radians(layout['orientations_deg'][pad])
    c, s = math.cos(a), math.sin(a)
    return px+c*x-s*y, py+s*x+c*y


def arms_family(config, layout, family):
    """chip9_geometry._arms with per-edge sizes. Returns arms in millimetres (Qiskit Metal units) and records."""
    arms, records = {}, []
    for route in link_centerlines(config, layout):
        line = route['line']; i, j = route['qubits']
        L = edge_link_params(config, family, i, j)
        middle = line.length/2; half_gap = L['gap_um']/2; length = L['paddle_length_um']
        if middle <= half_gap+length+L['lead_width_um']:
            raise ValueError(f'Physical link {i}-{j} is too short for the requested gap and paddles')
        central = substring(line, middle-half_gap-length, middle+half_gap+length)
        begin, end = np.array(central.coords[0]), np.array(central.coords[-1])
        if abs(np.linalg.norm(end-begin)-central.length) > 1e-7:
            raise ValueError(f'Physical link {i}-{j} needs a straight central segment across the coupling gap/paddles')
        angle = math.degrees(math.atan2(*(end-begin)[::-1]))
        pair = []
        for endpoint, distance, side in [(0, middle-half_gap-length/2, route['islands'][0]),
                                         (1, middle+half_gap+length/2, route['islands'][1])]:
            centre = line.interpolate(distance)
            paddle = translate(rotate(box(-length/2, -L['paddle_width_um']/2, length/2, L['paddle_width_um']/2), angle, origin=(0, 0)),
                               centre.x, centre.y)
            lead_line = substring(line, 0, distance) if endpoint == 0 else substring(line, distance, line.length)
            arm = unary_union([lead_line.buffer(L['lead_width_um']/2, cap_style=2, join_style=2), paddle])
            if arm.geom_type != 'Polygon' or not arm.is_valid: raise ValueError(f'Physical link {i}-{j} has an invalid arm')
            arms[f'link_{i}_{j}_{endpoint}'] = dict(label=f'q{route["qubits"][endpoint]}_{side}',
                                                     geometry=scale(arm, xfact=.001, yfact=.001, origin=(0, 0)))
            pair.append(paddle)
        gap = pair[0].distance(pair[1])
        if abs(gap-L['gap_um']) > 1e-7: raise ValueError(f'Physical link {i}-{j} does not realize the requested gap')
        records.append(dict(qubits=[i, j], islands=route['islands'], centreline_um=route['points_um'], link_um=L,
                            actual_paddle_gap_um=gap, gap_is_conductor=False, quantum_coupler_mode=False))
    return arms, records


def preview_polygons(config, family):
    """Pure-shapely conductor polygons (mm) matching chip_cad_family; used for checks and previews."""
    layout = resolve_layout(config)
    w, h, jg = (float(config[k]) for k in ('pad_width_um', 'pad_height_um', 'junction_gap_um'))
    terms = {}
    for q in range(layout['n_qubits']):
        a = layout['orientations_deg'][q]
        for name, sign in (('a', -1), ('b', 1)):
            r = translate(box(-w/2, -h/2, w/2, h/2), sign*(w+jg)/2, 0)
            r = translate(rotate(r, a, origin=(0, 0)), *layout['positions_um'][q])
            terms[f'q{q}_{name}'] = scale(r, xfact=.001, yfact=.001, origin=(0, 0))
    for kind in PORT_KINDS:
        for q in range(layout['n_qubits']):
            x, y, pw, pl, _ = port_local(config, family, q, kind)
            gx, gy = _to_global(layout, q, x, y)
            r = translate(rotate(box(-pw/2, -pl/2, pw/2, pl/2), layout['orientations_deg'][q], origin=(0, 0)), gx, gy)
            terms[f'{kind}{q}'] = scale(r, xfact=.001, yfact=.001, origin=(0, 0))
    arms, records = arms_family(config, layout, family)
    for name, spec in arms.items():
        joined = unary_union([terms[spec['label']], spec['geometry']])
        if joined.geom_type != 'Polygon': raise ValueError(f'Arm {name} disconnected from {spec["label"]}')
        terms[spec['label']] = joined
    labels = list(terms)
    for x, la in enumerate(labels):
        for lb in labels[x+1:]:
            if terms[la].intersects(terms[lb]): raise ValueError(f'Conductor short: {la}, {lb}')
    gaps = {f'{la}|{lb}': terms[la].distance(terms[lb])*1000 for x, la in enumerate(labels) for lb in labels[x+1:]}
    return terms, dict(min_gap_um=min(gaps.values()), records=records,
                       nearest=sorted(gaps.items(), key=lambda kv: kv[1])[:6])


def chain_request(n_data, family=None, *, spacing_um=1000., target_frequencies_GHz=None, name=''):
    """Straight chain of 2n-1 pads with the same link/pad/stub family as threeq_chain_cell.default_request."""
    n_pads = 2*n_data-1
    xs = [(k-(n_pads-1)/2)*spacing_um for k in range(n_pads)]
    geometry = normalize_chip9_geometry(dict(
        architecture=ARCHITECTURE, n_qubits=n_pads, topology='line', qubit_spacing_um=float(spacing_um),
        qubit_positions_um=[[round(x, 9), 0.] for x in xs], qubit_orientations_deg=[0.]*n_pads,
        ground_clearance_um=41., ground_precision_grid_mm=1e-9, mesh_algorithm_3d=10,
        ground_policy='boundary_connected_remove_enclosed_islands',
        capacitive_links=dict(lead_width_um=20., paddle_width_um=120., paddle_length_um=60., gap_um=30.),
        solver_order=2, solver_tolerance=1e-10, mesh_edge_scope='all', mesh_timeout_s=300., solver_timeout_s=1800.,
        max_rss_gib=3., threads=1, **MESH_PROFILES['p2_e12']))
    fam = normalize_family(family, n_pads)
    terms, checks = preview_polygons(geometry, fam)       # rejects shorts before any job
    if target_frequencies_GHz is None:
        target_frequencies_GHz = ([5.20, 6.74, 5.00, 6.94, 5.40] if n_data == 3 else
                                  [5.20, 6.74, 5.00, 6.94, 5.40, 6.84, 5.10, 7.04, 5.30])
    if len(target_frequencies_GHz) != n_pads: raise ValueError('One target frequency per mode required')
    return dict(schema='emulateqpu.chain_family_request.v1', name=name, n_data=n_data, mode_order=mode_order(n_data),
                geometry=geometry, family=fam, family_sha256=family_sha256(fam),
                quantum=dict(target_frequencies_GHz=[float(x) for x in target_frequencies_GHz],
                             junction_capacitance_fF=[0.]*n_pads, ncut=24),
                preview=dict(min_conductor_gap_um=checks['min_gap_um'], nearest_pairs_um=checks['nearest']),
                scope='Data transmons and physical coupler capacitors with declared lumped symmetric SQUIDs; electrostatics only')


def physical_signature(geometry, family):
    keep = {k: v for k, v in geometry.items() if not (k.startswith('mesh_') or k in (
        'solver_order', 'solver_tolerance', 'solver_timeout_s', 'max_rss_gib', 'threads', 'mesh_threads', 'solver_threads', 'mpi_ranks'))}
    return hashlib.sha256(json.dumps(dict(geometry=keep, family=family), sort_keys=True, allow_nan=False).encode()).hexdigest()


# ------------------------------------------------------------------ reduction (works for full or islands-only data)
def reduce_islands(C_II, C_IA, n_modes, junction_capacitance_fF=None):
    """Exactly circuit.reduce_capacitance's EC and port coefficients, from the two blocks they use.

    D has (+1, -1) on the two islands of each pad; EC = e^2/(2h) D^T C_II^-1 D in GHz; port-to-charge coefficients
    (C_II^-1 D)^T C_IA. Junction capacitance is added to the island block exactly as reduce_capacitance does.
    """
    C = np.array(C_II, float).copy()
    CJ = np.zeros(n_modes) if junction_capacitance_fF is None else np.asarray(junction_capacitance_fF, float)*1e-15
    scale_ = np.abs(C).max()
    if np.abs(C-C.T).max()/scale_ > 1e-6: raise ValueError('Nonreciprocal island block')
    C = (C+C.T)/2
    for q in range(n_modes):
        a, b = 2*q, 2*q+1
        C[a, a] += CJ[q]; C[b, b] += CJ[q]; C[a, b] -= CJ[q]; C[b, a] -= CJ[q]
    if np.linalg.eigvalsh(C)[0] <= 0: raise ValueError('Island block is not positive definite')
    D = np.zeros((2*n_modes, n_modes))
    for q in range(n_modes): D[2*q:2*q+2, q] = [1, -1]
    invD = np.linalg.solve(C, D)
    EC = (e*e/(2*h*1e9))*D.T@invD
    coeff = invD.T@np.asarray(C_IA, float)
    return EC, coeff
