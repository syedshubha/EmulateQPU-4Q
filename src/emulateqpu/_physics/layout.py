"""Physical layouts for the legacy five-transmon and physical nine-transmon modes.

Legacy graph edges are annotations. The explicit capacitive-link architecture
uses selected edges to build metal arms in chip9_geometry, retaining parasitics.
Coordinates are micrometres; rotations act on each qubit and both interfaces.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path

TOPOLOGIES = ("line", "ring", "star", "grid", "all-to-all", "custom")


def _number(value, label, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return value


def canonicalize_config(config):
    """Resolve public aliases and reject conflicting values rather than guess."""
    cfg = deepcopy(config)
    for canonical, alias in [("n_qubits", "nq"), ("qubit_spacing_um", "spacing_um"),
                             ("intended_edges", "custom_edges")]:
        if alias in cfg:
            if canonical in cfg and cfg[canonical] != cfg[alias]:
                raise ValueError(f"Conflicting {canonical} and {alias}")
            cfg[canonical] = cfg.pop(alias)
    cfg["topology"] = str(cfg.get("topology", "line")).replace("_", "-").lower()
    return cfg


def resolve_layout(config):
    """Return explicit physical positions and a separately declared target graph."""
    cfg = canonicalize_config(config)
    n_float = _number(cfg.get("n_qubits", 2), "n_qubits", positive=True)
    n = int(n_float)
    maximum = 9 if cfg.get('architecture') == 'fixed_direct_capacitive_links' else 5
    if n != n_float or not 1 <= n <= maximum:
        raise ValueError(f"n_qubits must be an integer from 1 through {maximum}")
    topology = cfg["topology"]
    if topology not in TOPOLOGIES:
        raise ValueError(f"Unknown topology {topology!r}; choose {TOPOLOGIES}")
    pitch = _number(cfg.get("qubit_spacing_um", 1200), "qubit_spacing_um", positive=True)
    grid = None
    if "qubit_positions_um" in cfg and cfg["qubit_positions_um"] is not None:
        raw = cfg["qubit_positions_um"]
        if not isinstance(raw, (list, tuple)) or len(raw) != n:
            raise ValueError("qubit_positions_um must contain one [x,y] pair per qubit")
        positions = []
        for i, xy in enumerate(raw):
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                raise ValueError(f"Position q{i} must be an [x,y] pair in um")
            positions.append([_number(xy[0], f"q{i} x"), _number(xy[1], f"q{i} y")])
        source = "explicit positions"
    else:
        if topology == "custom":
            raise ValueError("Custom topology requires explicit qubit_positions_um")
        if topology == "line" or n <= 2:
            positions = [[(i - (n - 1) / 2) * pitch, 0.0] for i in range(n)]
        elif topology in ("ring", "all-to-all"):
            radius = pitch / (2 * math.sin(math.pi / n))
            positions = [[radius * math.cos(2 * math.pi * i / n),
                          radius * math.sin(2 * math.pi * i / n)] for i in range(n)]
        elif topology == "star":
            positions = [[0.0, 0.0]] + [[pitch * math.cos(2 * math.pi * i / (n - 1)),
                                         pitch * math.sin(2 * math.pi * i / (n - 1))]
                                        for i in range(n - 1)]
        else:
            cols = math.ceil(math.sqrt(n))
            grid = [(i % cols, i // cols) for i in range(n)]
            cx = (max(x for x, y in grid) + min(x for x, y in grid)) / 2
            cy = (max(y for x, y in grid) + min(y for x, y in grid)) / 2
            positions = [[(x - cx) * pitch, (y - cy) * pitch] for x, y in grid]
        source = f"generated {topology} placement"
    if len({tuple(xy) for xy in positions}) != n:
        raise ValueError("Qubit positions must be distinct")
    angles = cfg.get("qubit_orientations_deg", [0.0] * n)
    if not isinstance(angles, (list, tuple)) or len(angles) != n:
        raise ValueError("qubit_orientations_deg must contain one angle per qubit")
    angles = [_number(x, f"q{i} orientation") % 360 for i, x in enumerate(angles)]
    if "intended_edges" in cfg:
        raw_edges = cfg["intended_edges"]
    elif topology == "all-to-all":
        raw_edges = list(itertools.combinations(range(n), 2))
    elif topology == "star":
        raw_edges = [(0, i) for i in range(1, n)]
    elif topology == "custom":
        raw_edges = []
    elif topology == "grid":
        cols = math.ceil(math.sqrt(n))
        grid = [(i % cols, i // cols) for i in range(n)]
        raw_edges = [(i, j) for i, j in itertools.combinations(range(n), 2)
                     if abs(grid[i][0] - grid[j][0]) + abs(grid[i][1] - grid[j][1]) == 1]
    else:
        raw_edges = [(i, i + 1) for i in range(n - 1)]
        if topology == "ring" and n > 2:
            raw_edges.append((0, n - 1))
    if not isinstance(raw_edges, (list, tuple)):
        raise ValueError("intended_edges must be a list of qubit index pairs")
    edges = []
    for edge in raw_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError("Each intended edge must contain two qubit indices")
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n for i in edge):
            raise ValueError("Edge indices must be integer qubit indices in range")
        pair = tuple(sorted(edge))
        if pair[0] == pair[1] or list(pair) in edges:
            raise ValueError("Self edges and duplicate intended edges are invalid")
        edges.append(list(pair))
    return dict(n_qubits=n, topology=topology, positions_um=positions,
                orientations_deg=angles, intended_edges=sorted(edges), placement_source=source,
                coupling_architecture=cfg.get('architecture', 'fixed_direct_capacitive'),
                edge_semantics=("Selected edges realize separate capacitive metal arms in CAD; coupling strength is extracted, not set by a graph weight. All parasitic pairs remain." if cfg.get("architecture") == "fixed_direct_capacitive_links" else "Logical target annotation only; no engineered couplers or equal edges. All electrostatic pair couplings are retained."))


def footprint_bounds_um(config, layout=None):
    """Bounds of all pad and drive/readout rectangles, including rotations."""
    cfg = canonicalize_config(config)
    layout = layout or resolve_layout(cfg)
    names = ["pad_width_um", "pad_height_um", "junction_gap_um", "junction_width_um",
             "drive_gap_um", "drive_width_um", "drive_length_um", "readout_gap_um",
             "readout_width_um", "readout_length_um", "ground_clearance_um"]
    values = {k: _number(cfg[k], k, positive=True) for k in names}
    w, h, gap = (values[k] for k in ["pad_width_um", "pad_height_um", "junction_gap_um"])
    local = [(-(w + gap)/2, 0, w, h), ((w + gap)/2, 0, w, h),
             (-(w + gap)/2, -h/2-values["drive_gap_um"]-values["drive_length_um"]/2,
              values["drive_width_um"], values["drive_length_um"]),
             ((w + gap)/2, h/2+values["readout_gap_um"]+values["readout_length_um"]/2,
              values["readout_width_um"], values["readout_length_um"])]
    points = []
    for (px, py), angle in zip(layout["positions_um"], layout["orientations_deg"]):
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        for x, y, rw, rh in local:
            for dx, dy in itertools.product([-rw/2, rw/2], [-rh/2, rh/2]):
                points.append((px+c*(x+dx)-s*(y+dy), py+s*(x+dx)+c*(y+dy)))
    if cfg.get('architecture') == 'fixed_direct_capacitive_links':
        from .chip9_geometry import route_footprint_points
        points.extend(route_footprint_points(cfg, layout))
    return [min(x for x,y in points), min(y for x,y in points),
            max(x for x,y in points), max(y for x,y in points)]


def make_layout_config(topology=None, n_qubits=None, *, config=None, **overrides):
    """Complete EM config with automatic chip size unless dimensions are explicit.

    Generated layout positions stay parametric; resolve_layout supplies preview
    coordinates. An explicit qubit_positions_um overrides generated placement.
    """
    base = json.loads((Path(__file__).parent / "configs/em_baseline.json").read_text())
    supplied = dict(config or {})
    supplied.update(overrides)
    # A caller's alias overrides a default without creating an artificial clash.
    supplied = canonicalize_config(supplied)
    if config is None and topology is None and "topology" not in overrides:
        supplied["topology"] = "line"
    base.update(supplied)
    if topology is not None:
        base["topology"] = str(topology).replace("_", "-")
    if n_qubits is not None:
        base["n_qubits"] = n_qubits
    base = canonicalize_config(base)
    layout = resolve_layout(base)
    bounds = footprint_bounds_um(base, layout)
    margin = max(300.0, 2*float(base["ground_clearance_um"]))
    for dim, lo, hi in [("chip_width_um", bounds[0], bounds[2]),
                        ("chip_height_um", bounds[1], bounds[3])]:
        if dim not in supplied:
            base[dim] = math.ceil(2*(max(abs(lo), abs(hi))+margin)/100)*100.0
    return base


def geometry_signature(config):
    """Hash physical CAD/material inputs, including realized capacitive links.

    Legacy graph-only annotations remain excluded from this physical identity.

    Meshing and solver settings also remain bound; this is conservative for UI
    replay safety. Equivalent generated and explicit positions hash identically.
    """
    cfg = canonicalize_config(config)
    layout = resolve_layout(cfg)
    graph_keys = ["topology", "custom_edges", "qubit_positions_um", "qubit_orientations_deg", "qubit_spacing_um"]
    if cfg.get('architecture') == 'fixed_direct_capacitive_links':
        from .chip9_geometry import normalized_links, numerical_profile
        grid, algorithm = numerical_profile(cfg)
        if grid == 0.:
            cfg.pop('ground_precision_grid_mm', None)
        if algorithm == 1:
            cfg.pop('mesh_algorithm_3d', None)
        cfg['capacitive_links'] = normalized_links(cfg, layout)
        cfg['intended_edges'] = layout['intended_edges']
    else:
        graph_keys.append('intended_edges')
    for key in graph_keys:
        cfg.pop(key, None)
    cfg["qubit_positions_um"] = layout["positions_um"]
    cfg["qubit_orientations_deg"] = layout["orientations_deg"]
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()
