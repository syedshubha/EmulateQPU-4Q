"""Explicit linear-chain constructors; no fields, calibration, or dynamics on import.

The circuit fixture is assumed. Optional physical CAD uses the existing
capacitive-link polygon helpers with an explicit generalized line placement.
No arbitrary placement or spatial-accuracy claim follows from construction.
"""
from __future__ import annotations

from copy import deepcopy
import math


def mode_order(n_data):
    if type(n_data) is not int or n_data < 2:
        raise ValueError('A linear chain requires an integer n_data >= 2')
    result = []
    for k in range(n_data):
        result.append(f'q{k}')
        if k < n_data-1:
            result.append(f'c{k}{k+1}' if k < 9 else f'c{k}_{k+1}')
    return result


def linear_chain_cell(n_data=6, *, ncut=24, coupling_GHz=.001, name=None):
    """Assumed full signed EC/EJ circuit with n data and n-1 explicit couplers.

    Couplings are circuit parameters, not capacitance extracted from the CAD
    constructor. Every old explicit circuit and default engine identity remains
    unchanged. This fixture never retargets a supplied extraction's junctions.
    """
    labels = mode_order(n_data); m = len(labels)
    if not isinstance(coupling_GHz, (int, float)) or isinstance(coupling_GHz, bool) or not math.isfinite(coupling_GHz) or abs(coupling_GHz) > .02:
        raise ValueError('Assumed nearest-neighbor charging coupling must be finite with magnitude <= .02 GHz')
    if type(ncut) is not int or not 16 <= ncut <= 48:
        raise ValueError('Charge cutoff must be an integer 16..48')
    roles = ['data' if i % 2 == 0 else 'coupler' for i in range(m)]
    ec = [[0. for _ in labels] for _ in labels]
    for i in range(m):
        ec[i][i] = .25+.005*(i//2 % 3) if roles[i] == 'data' else .3
        if i:
            ec[i][i-1] = ec[i-1][i] = float(coupling_GHz)
    return dict(schema='emulateqpu.chain.cell.v1', name=name or f'assumed_{n_data}_data_linear_chain',
        mode_labels=labels, mode_roles=roles, EC_matrix_GHz=ec,
        EJ_idle_GHz=[14.+.2*(i//2 % 4) if roles[i] == 'data' else 9.+.1*(i//2 % 3) for i in range(m)],
        idle_flux_Phi0=[0. if r == 'data' else .2 for r in roles], ncut=ncut,
        lines={label: dict(tau_ns=.15, gain=1., settling_time_constants=5., rise_fraction=.2)
               for label, role in zip(labels, roles) if role == 'coupler'},
        parameter_origin='assumed_circuit',
        provenance=dict(kind='explicit assumed finite-model fixture', geometry_extracted=False,
                        topology='linear', edges=[[labels[i], labels[i+1]] for i in range(m-1)],
                        port_mapping={label: {'drive': f'drive{i}', 'readout': f'readout{i}'} for i, label in enumerate(labels)}),
        assumptions=dict(closed_system=True, calibrated_gate=False, physical_validation=False,
                         scope='Selected-input finite-model software demonstration; truncation is not physically converged'))


def linear_chain_geometry(n_data=6, *, family=None, spacing_um=1000., target_frequencies_GHz=None, max_modes=129):
    """Construct and check actual planar conductor polygons for a straight family.

    Returns ``(request, conductors)``; conductor geometries use millimeters.
    It reuses the frozen family's port, link-arm, and junction-pad dimensions.
    Existing three/five-data requests delegate unchanged. New sizes have their
    own placement identity and explicit terminal mapping. No mesh or solver is
    launched. The optional CAD dependency profile is required.

    This generalized polygon path is separate from historical field campaign
    controllers whose 3/5-data assumptions remain frozen.
    """
    labels = mode_order(n_data); m = len(labels)
    if type(max_modes) is not int or max_modes < 3 or m > max_modes:
        raise ValueError('Linear CAD exceeds the declared max_modes construction limit')
    if isinstance(spacing_um, bool) or not math.isfinite(float(spacing_um)) or float(spacing_um) <= 0:
        raise ValueError('spacing_um must be finite and positive')
    try:
        from emulateqpu._physics import chain_family as cf
        from shapely.affinity import translate, scale
        from shapely.geometry import box
        from shapely.ops import unary_union
    except ImportError as exc:
        raise RuntimeError('Linear CAD construction requires the optional physical dependencies') from exc
    if n_data in (3, 5):
        request = cf.chain_request(n_data, family, spacing_um=spacing_um,
                                   target_frequencies_GHz=target_frequencies_GHz)
        conductors, _ = cf.preview_polygons(request['geometry'], request['family'])
        return request, conductors
    # Reuse completed canonical dimensions/defaults, then state the new linear
    # placement directly instead of invoking the frozen max-nine layout resolver.
    reference = cf.chain_request(3, spacing_um=spacing_um)
    config = deepcopy(reference['geometry'])
    positions = [[round((i-(m-1)/2)*float(spacing_um), 9), 0.] for i in range(m)]
    edges = [[i, i+1] for i in range(m-1)]
    config.update(n_qubits=m, topology='line', qubit_positions_um=positions,
                  qubit_orientations_deg=[0.]*m, intended_edges=edges)
    config['capacitive_links']['edges'] = edges
    config['capacitive_links']['routes'] = {}
    layout = dict(n_qubits=m, topology='line', positions_um=positions,
                  orientations_deg=[0.]*m, intended_edges=edges)
    fam = cf.normalize_family(family, m)
    w, h, gap = (float(config[k]) for k in ('pad_width_um', 'pad_height_um', 'junction_gap_um'))
    conductors = {}
    for i, (px, py) in enumerate(positions):
        for side, sign in (('a', -1), ('b', 1)):
            conductors[f'q{i}_{side}'] = scale(translate(box(-w/2, -h/2, w/2, h/2), px+sign*(w+gap)/2, py),
                                               xfact=.001, yfact=.001, origin=(0, 0))
        for kind in cf.PORT_KINDS:
            x, y, pw, pl, _ = cf.port_local(config, fam, i, kind)
            conductors[f'{kind}{i}'] = scale(translate(box(-pw/2, -pl/2, pw/2, pl/2), px+x, py+y),
                                             xfact=.001, yfact=.001, origin=(0, 0))
    arms, records = cf.arms_family(config, layout, fam)
    for key, arm in arms.items():
        joined = unary_union([conductors[arm['label']], arm['geometry']])
        if joined.geom_type != 'Polygon' or not joined.is_valid:
            raise ValueError(f'Physical link arm {key} is disconnected or invalid')
        conductors[arm['label']] = joined
    keys = list(conductors); min_gap = float('inf')
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            if conductors[a].intersects(conductors[b]):
                raise ValueError(f'Conductor short: {a}, {b}')
            min_gap = min(min_gap, 1000*conductors[a].distance(conductors[b]))
    bounds = unary_union(list(conductors.values())).bounds
    margin = max(300., 2*float(config['ground_clearance_um']))
    config['chip_width_um'] = math.ceil(2*(1000*max(abs(bounds[0]), abs(bounds[2]))+margin)/100)*100.
    config['chip_height_um'] = math.ceil(2*(1000*max(abs(bounds[1]), abs(bounds[3]))+margin)/100)*100.
    if target_frequencies_GHz is None:
        target_frequencies_GHz = [5.2+.1*(i//2 % 4) if i % 2 == 0 else 6.74+.1*(i//2 % 4) for i in range(m)]
    if len(target_frequencies_GHz) != m or any(isinstance(v, bool) or not math.isfinite(float(v)) or v <= 0 for v in target_frequencies_GHz):
        raise ValueError('One finite positive target frequency per mode required')
    request = dict(schema='emulateqpu.linear_chain_geometry.v1', n_data=n_data, mode_order=labels,
        geometry=config, family=fam, family_sha256=cf.family_sha256(fam),
        quantum=dict(target_frequencies_GHz=list(map(float, target_frequencies_GHz)), junction_capacitance_fF=[0.]*m, ncut=24),
        terminal_mapping={label: {'islands': [f'q{i}_a', f'q{i}_b'], 'drive': f'drive{i}', 'readout': f'readout{i}'}
                          for i, label in enumerate(labels)},
        preview=dict(min_conductor_gap_um=min_gap, links=records, conductors=len(conductors)),
        scope='Validated linear conductor CAD; no extraction, arbitrary-placement support, or numerical field qualification')
    return request, conductors
