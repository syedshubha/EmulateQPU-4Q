"""Small-signal driven modal response and weak-excitation spectrum."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..io import digest, file_hash, read_json
from .reduction import MODEL_SCHEMA, QUALIFICATION_SCHEMA


def load_model(directory):
    """Open a qualified model with its original reduction report and checksums."""
    directory = Path(directory)
    manifest = read_json(directory / "artifact.json")
    if (manifest.get("schema") != "emulateqpu.fullwave.artifact.v1" or manifest.get("passed") is not True
            or set(manifest["files_sha256"]) != {"qualification.json", "electrostatic-cell.json", "model.json"}):
        raise ValueError("A passing full-wave artifact is required")
    for name, expected in manifest["files_sha256"].items():
        path = directory / name
        if path.is_symlink() or file_hash(path) != expected:
            raise ValueError(f"Full-wave artifact integrity mismatch for {name}")
    report, model = (read_json(directory / name) for name in ("qualification.json", "model.json"))
    if (report.get("schema") != QUALIFICATION_SCHEMA or report.get("passed") is not True
            or model.get("schema") != MODEL_SCHEMA or model.get("qualification_sha256") != digest(report)
            or model.get("design_sha256") != report.get("design_sha256")):
        raise ValueError("Full-wave model is detached from its qualification")
    return model


def fock_energy_GHz(model, occupations):
    """Return E/h above vacuum for up to two excitations in the first-order model."""
    if (len(occupations) != len(model["mode_ids"]) or any(type(n) is not int or n < 0 for n in occupations)
            or sum(occupations) > 2):
        raise ValueError("Expected one nonnegative integer per mode, at most two total excitations")
    n = np.asarray(occupations)
    f, kerr = np.asarray(model["f01_GHz"]), np.asarray(model["kerr_GHz"])
    return float(f @ n + 0.5 * (n @ kerr @ n - np.diag(kerr) @ n))


def drive_response(model, port, frequency_GHz, duration_ns, incident_photons_per_ns):
    """Approximate a square incident tone with independent damped modes in weak drive."""
    ports = [item["label"] for item in model["ports"]]
    if port not in ports:
        raise ValueError(f"Unknown drive port {port}")
    values = (frequency_GHz, duration_ns, incident_photons_per_ns)
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
        raise ValueError("Frequency, duration and incident intensity must be finite numbers")
    grid = np.asarray(model["frequency_grid_GHz"])
    if not (grid[0] <= frequency_GHz <= grid[-1] and duration_ns > 0 and incident_photons_per_ns >= 0):
        raise ValueError("Use a tone inside the qualified frequency band with positive duration and nonnegative intensity")
    index = ports.index(port)
    scattering = np.asarray(model["scattering_real"]) + 1j * np.asarray(model["scattering_imag"])
    rows = scattering[:, :, index].T
    values = np.asarray([np.interp(frequency_GHz, grid, row.real) + 1j * np.interp(frequency_GHz, grid, row.imag)
                         for row in rows])
    kappa = np.abs(np.asarray(model["kappa_signed_GHz"])) * (2 * np.pi)
    f = np.asarray(model["linear_mode_GHz"])
    decay = 0.5 * kappa.sum(axis=1) + 2j * np.pi * (f - frequency_GHz)
    accumulation = np.full(decay.shape, float(duration_ns), dtype=complex)
    nonzero = np.abs(decay) > 0
    accumulation[nonzero] = -np.expm1(-decay[nonzero] * duration_ns) / decay[nonzero]
    population = np.abs(np.sqrt(kappa[:, index] * incident_photons_per_ns) * accumulation)**2
    if np.any(population > .1):
        raise ValueError("Requested drive exceeds the 0.1 mean-photon weak-excitation range")
    return {"schema": "emulateqpu.fullwave.response.v1", "input_port": port, "frequency_GHz": frequency_GHz,
            "duration_ns": duration_ns, "incident_photons_per_ns": incident_photons_per_ns,
            "mode_ids": model["mode_ids"], "mode_population_at_tone_off": population.tolist(),
            "output_power_fraction": {p: float(v) for p, v in zip(ports, np.abs(values)**2)},
            "scope": "Independent damped harmonic modes and separately interpolated linear scattering",
            "synthetic": model["synthetic"], "physical_device_validated": False}
