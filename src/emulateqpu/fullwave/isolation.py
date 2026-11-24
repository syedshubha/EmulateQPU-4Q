"""Compare passive RF power transport at matched drive reference planes."""
from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np

from ..io import digest, file_hash, read_json
from .design import ASSEMBLY_PORT_ORDER, DATA_QUBITS, DRIVE_PORTS, PORT_ORDER, SHARED_PORT_ORDER
from .reduction import ASSEMBLY_SHARED_PAIR, TOLERANCE
from .response import load_model


ISOLATION_SCHEMA = "emulateqpu.fullwave.isolation.v1"


def _numeric_array(values):
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("Expected finite numeric RF values")
    return array.astype(float, copy=False)


def _port_layout(model):
    ports = model["ports"]
    labels = tuple(port["label"] for port in ports)
    assembly = labels == ASSEMBLY_PORT_ORDER
    topology = model.get("readout_topology", "dedicated")
    expected = {"dedicated": PORT_ORDER, "shared": SHARED_PORT_ORDER}.get(topology)
    if expected is None or (not assembly and labels != expected):
        raise ValueError("Expected the four-drive port inventory of a dedicated or shared four-readout layout")
    if any(type(port["port"]) is not int or port["port"] != number or port["Z_ref_ohm"] != 50
           for number, port in enumerate(ports, 1)):
        raise ValueError("Expected ordered 50-ohm RF reference ports")

    if topology == "shared":
        shared_pair = ASSEMBLY_SHARED_PAIR if assembly else SHARED_PORT_ORDER[len(DRIVE_PORTS):]
        expected_channels = [(qubit, *shared_pair) for qubit in DATA_QUBITS]
    else:
        expected_channels = [(qubit, f"readout_{qubit}_in", f"readout_{qubit}_out") for qubit in DATA_QUBITS]
    channels = model.get("readout_channels", ())
    if [(row["qubit"], row["input_port"], row["output_port"]) for row in channels] != expected_channels:
        raise ValueError("Expected the four physically mapped readout channels of the selected topology")
    readout_ports = list(dict.fromkeys(port for _, first, last in expected_channels for port in (first, last)))

    if assembly:
        connected = set(DRIVE_PORTS) | set(readout_ports) | set(ASSEMBLY_PORT_ORDER[len(PORT_ORDER):])
        if any(port.get("active") is not (port["label"] in connected) for port in ports):
            raise ValueError("Carrier port activity differs from the connected die endpoints")
    reference_planes = _numeric_array([port["reference_plane_mm"] for port in ports])
    if reference_planes.shape != (len(labels), 3):
        raise ValueError("Expected one three-coordinate reference plane per port")
    plane_kind = "carrier" if assembly else "die"
    matched_planes = reference_planes if assembly else reference_planes[:len(DRIVE_PORTS)]
    return labels, topology, readout_ports, plane_kind, matched_planes


def _load_artifact(directory):
    manifest_path = directory / "artifact.json"
    manifest_sha = file_hash(manifest_path)
    model = load_model(directory)
    qualification = read_json(directory / "qualification.json")
    if model.get("synthetic") is not False or qualification.get("synthetic") is not False:
        raise ValueError("Isolation comparison requires non-synthetic full-wave qualification")
    if model["qualification_sha256"] != digest(qualification) or manifest_sha != file_hash(manifest_path):
        raise ValueError("Full-wave artifact changed during isolation comparison")
    labels, topology, readout_ports, plane_kind, planes = _port_layout(model)
    grid = _numeric_array(model["frequency_grid_GHz"])
    if grid.ndim != 1 or len(grid) < 2 or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
        raise ValueError("Expected a strictly increasing positive frequency grid")
    real = _numeric_array(model["scattering_real"])
    imag = _numeric_array(model["scattering_imag"])
    shape = (len(grid), len(labels), len(labels))
    if real.shape != shape or imag.shape != shape:
        raise ValueError("Scattering table shape differs from the frequency grid or port inventory")
    magnitude = np.hypot(real, imag)
    if np.any(magnitude > 1 + TOLERANCE["scattering_passivity_abs"]):
        raise ValueError("Scattering amplitude exceeds the passive RF tolerance")
    entry = {"artifact_path": str(directory), "artifact_manifest_sha256": manifest_sha,
             "design_sha256": model["design_sha256"], "qualification_sha256": model["qualification_sha256"],
             "readout_topology": topology, "reference_plane_kind": plane_kind,
             "readout_ports": readout_ports}
    return entry, grid, magnitude**2, labels, planes


def _drive_metrics(frequencies, powers, labels, readout_ports):
    readout_indices = [labels.index(label) for label in readout_ports]
    drives = {}
    for drive_index, drive in enumerate(DRIVE_PORTS):
        emitted = powers[:, :, drive_index]
        spectators = {label: emitted[:, index] for index, label in enumerate(DRIVE_PORTS) if label != drive}
        to_readout = emitted[:, readout_indices].sum(axis=1)
        to_other_drives = np.sum(tuple(spectators.values()), axis=0)
        readout_max = int(to_readout.argmax())
        other_drives_max = int(to_other_drives.argmax())
        drives[drive] = {
            "readout_power_fraction": to_readout.tolist(),
            "reflected_power_fraction": emitted[:, drive_index].tolist(),
            "spectator_drive_power_fraction": {name: values.tolist() for name, values in spectators.items()},
            "other_drive_power_fraction": to_other_drives.tolist(),
            "max_readout_power_fraction": float(to_readout[readout_max]),
            "max_readout_frequency_GHz": float(frequencies[readout_max]),
            "max_other_drive_power_fraction": float(to_other_drives[other_drives_max]),
            "max_other_drive_frequency_GHz": float(frequencies[other_drives_max]),
        }
    return drives


def compare_isolation(artifacts, *, band_GHz=None):
    """Compare sampled power fractions for two or more qualified layout artifacts."""
    if isinstance(artifacts, (str, bytes, Path)):
        raise ValueError("Provide two or more distinct qualified artifact directories")
    try:
        directories = [Path(path).resolve() for path in artifacts]
    except (OSError, TypeError) as error:
        raise ValueError("Provide two or more distinct qualified artifact directories") from error
    if len(directories) < 2 or len(set(directories)) != len(directories):
        raise ValueError("Provide two or more distinct qualified artifact directories")
    band = None if band_GHz is None else _numeric_array(band_GHz)
    if band is not None and (band.shape != (2,) or not 0 < band[0] < band[1]):
        raise ValueError("Frequency band must have two positive, increasing bounds in GHz")

    inputs = []
    for directory in directories:
        try:
            inputs.append(_load_artifact(directory))
        except (OSError, AttributeError, KeyError, OverflowError, TypeError) as error:
            raise ValueError(f"Invalid qualified full-wave artifact in {directory}") from error
    first, _, _, _, first_planes = inputs[0]
    if len({item[0]["design_sha256"] for item in inputs}) != len(inputs):
        raise ValueError("Isolation comparison requires distinct full-wave designs")
    if any(model["reference_plane_kind"] != first["reference_plane_kind"] or not np.array_equal(planes, first_planes)
           for model, _, _, _, planes in inputs[1:]):
        raise ValueError("Isolation comparison requires matched 50-ohm reference planes")
    frequencies = reduce(np.intersect1d, (grid for _, grid, _, _, _ in inputs))
    if band is not None:
        frequencies = frequencies[(frequencies >= band[0]) & (frequencies <= band[1])]
    if not len(frequencies):
        raise ValueError("No exactly matching solved frequencies in the comparison band")
    for model, grid, powers, labels, _ in inputs:
        powers = powers[np.searchsorted(grid, frequencies)]
        model["drives"] = _drive_metrics(frequencies, powers, labels, model["readout_ports"])

    return {
        "schema": ISOLATION_SCHEMA,
        "frequency_grid_GHz": frequencies.tolist(),
        "requested_band_GHz": None if band is None else band.tolist(),
        "drive_ports": list(DRIVE_PORTS),
        "models": [model for model, _, _, _, _ in inputs],
        "scope": "Sampled linear RF power at matched reference planes with one driven port and matched loads. "
                 "Readout assignment, quantum crosstalk, and device security require separate evidence.",
    }
