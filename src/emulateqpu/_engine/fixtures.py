"""Assumed finite-circuit fixtures and model construction."""
from copy import deepcopy
from importlib.resources import files
import json

import numpy as np


def demo_cell() -> dict:
    """Read the small assumed two-data/one-coupler circuit; no computation."""
    return json.loads(files('emulateqpu').joinpath('examples/two_data_engineering.json').read_text())


def demo_schedule() -> dict:
    """Read a four-ns uncalibrated microwave schedule; no computation."""
    return json.loads(files('emulateqpu').joinpath('examples/two_data_microwave.json').read_text())


def build_model(cell: dict, *, dims, cap=None, encoding_method='auto', cap_exempt=()):
    """Build the supplied finite cosine/charge Hamiltonian with all signed EC.

    One explicit retained dimension is required per mode. An excitation cap is
    a numerical truncation, not a convergence certificate. No EJ is retargeted.
    """
    from .model import build_model as original_build
    dims = tuple(dims)
    if not dims or any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 1 for d in dims):
        raise ValueError('dims must contain explicit positive integer retained dimensions')
    if cap is not None and (isinstance(cap, (bool, np.bool_)) or not isinstance(cap, (int, np.integer)) or cap < 0):
        raise ValueError('cap must be None or a nonnegative integer')
    return original_build(deepcopy(cell), dims, cap, encoding_method=encoding_method, cap_exempt=cap_exempt)
