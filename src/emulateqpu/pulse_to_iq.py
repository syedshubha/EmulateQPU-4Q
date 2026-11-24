"""Run an assumed four-channel pulse-to-IQ example in a fresh output directory.

Run with ``python -m emulateqpu.pulse_to_iq /path/to/new-directory``.
The receiver treats the four output bands as independent. The parameters
exercise the software with a weak readout and an explicit microwave pulse.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from . import example, run_request
from ._engine.measurement import qnd_instrument
from .api import load_model
from .io import atomic_json, file_hash


QUBITS = ("q0", "q1", "q2", "q3")
PREPARED_STATES = ("0000", "1000", "0100", "0010", "0001")
OUTCOMES = {f"{value:04b}" for value in range(16)}
PROBABILITY_TOLERANCE = 1e-7


def _readout_config():
    channels = []
    for qubit, frequency in zip(QUBITS, (6.45, 6.65, 6.85, 7.05), strict=True):
        channels.append({
            "name": f"readout_{qubit}",
            "reads": qubit,
            "readout_frequency_GHz": frequency,
            "filter_frequency_GHz": frequency,
            "drive_frequency_GHz": frequency,
            "coupling_MHz": 22.0,
            "filter_kappa_per_ns": 0.095,
            "drive_amplitude_sqrt_photons_per_ns": 0.13,
            "efficiency": 0.5,
            "chi_MHz": {qubit: 4.0},
        })
    return {
        "model": "dispersive_qnd_heterodyne",
        "basis": "local_levels",
        "duration_ns": 36.0,
        "drive_duration_ns": 24.0,
        "response_order": 16,
        "max_photons": 1.0,
        "channels": channels,
    }


def _run(output, relative_path, base, readout, label, outcome, schedule=None):
    request = deepcopy(base)
    request["initial"]["labels"] = [label]
    if schedule is not None:
        request["schedule"] = schedule
    request["measurement"] = {
        "qubits": list(QUBITS),
        "instrument": {**readout, "outcome": outcome},
    }
    directory = output / relative_path
    result = run_request(request, directory)
    if result["status"] != "PASS" or not result["complete"]:
        raise RuntimeError(f"Calculation did not pass for {relative_path}")
    result_path = directory / "result.json"
    evidence = {"result": str(result_path.relative_to(output)), "sha256": file_hash(result_path)}
    if outcome == "record":
        record = result["trace"][-1]["instrument"][0]
        evidence.update(iq=record["record"], seed=record["seed"], log_likelihood=record["log_likelihood"])
        return evidence

    summary = result["summary"]
    probabilities = dict(zip(
        (entry["measurement"] for entry in summary["measurement_records"]),
        summary["branch_probabilities"],
        strict=True,
    ))
    if (set(probabilities) != OUTCOMES or not all(value >= 0 for value in probabilities.values())
            or abs(sum(probabilities.values()) - 1) > PROBABILITY_TOLERANCE):
        raise RuntimeError(f"Invalid outcome probabilities for {relative_path}")
    evidence["outcome_probabilities"] = probabilities
    evidence["reported_one_probability"] = {
        qubit: sum(probability for bits, probability in probabilities.items() if bits[i] == "1")
        for i, qubit in enumerate(QUBITS)
    }
    evidence["q0_level_one_probability"] = summary["bare_level_populations"]["q0"][1]
    return evidence


def _centroids(model, config):
    instrument, _ = qnd_instrument(model, config)
    row_by_label = {tuple(levels): row for row, levels in enumerate(instrument.levels)}
    rows = [row_by_label[tuple(map(int, state))] for state in PREPARED_STATES]
    return instrument.observed[rows], instrument.metadata


def run_example(output):
    """Save a deterministic prepared-state benchmark and a matched pulse control."""
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True)
    fixture = example("four-qubit")
    fixture["name"] = "four-channel-pulse-to-iq"
    fixture["purpose"] = "Assumed four-qubit control and independent-band heterodyne software benchmark."
    fixture["model"]["encoding"]["labels"] = list(PREPARED_STATES)
    fixture["schedule"] = {"duration_ns": 0.0, "drives": [], "flux": []}
    readout = _readout_config()

    prepared = {}
    for state in PREPARED_STATES:
        prepared[state] = {
            "classified": _run(output, f"prepared/{state}/classified", fixture, readout, state, "classified"),
            "record": _run(output, f"prepared/{state}/record", fixture, readout, state, "record"),
        }
    replay = _run(output, "record-replay", fixture, readout, "0000", "record")
    baseline = prepared["0000"]["record"]

    model = load_model(output / "prepared" / "0000" / "classified")
    centroids, receiver = _centroids(model, readout)
    distances = np.abs(centroids[1:] - centroids[0])
    coupling_control = deepcopy(readout)
    for channel in coupling_control["channels"]:
        channel["chi_MHz"][channel["reads"]] = 0.0
    null_centroids, _ = _centroids(model, coupling_control)
    null_distance = float(np.max(np.abs(null_centroids - null_centroids[0])))

    idle = {"duration_ns": 7.0, "drives": [], "flux": []}
    pulse = {**idle, "drives": [{
        "mode": "q0",
        "start_ns": 0.0,
        "duration_ns": idle["duration_ns"],
        "amplitude_GHz": 0.05,
        "frequency_GHz": model.analysis["local_f01_GHz"][model.data_modes[0]],
        "phase_rad": 0.0,
        "envelope": "cosine",
    }]}
    pulsed = _run(output, "control/q0-pulse", fixture, readout, "0000", "classified", pulse)
    matched_idle = _run(output, "control/matched-idle", fixture, readout, "0000", "classified", idle)
    excitation_change = pulsed["q0_level_one_probability"] - matched_idle["q0_level_one_probability"]
    readout_change = pulsed["reported_one_probability"]["q0"] - matched_idle["reported_one_probability"]["q0"]

    ground_marginals = prepared["0000"]["classified"]["reported_one_probability"]
    prepared_readout_changes = {
        qubit: prepared[state]["classified"]["reported_one_probability"][qubit] - ground_marginals[qubit]
        for qubit, state in zip(QUBITS, PREPARED_STATES[1:], strict=True)
    }
    checks = {
        "addressed_centroids_separate": bool(np.all(np.diag(distances) > 0.05)),
        "zero_chi_centroids_coincide": null_distance < 1e-12,
        "prepared_excitations_increase_addressed_reports": min(prepared_readout_changes.values()) > 0.01,
        "record_seed_reproduces_iq_and_likelihood": all(baseline[k] == replay[k] for k in ("seed", "iq", "log_likelihood")),
        "microwave_pulse_changes_state_and_report": excitation_change > 0.05 and readout_change > 0.01,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "parameter_origin": "assumed circuit and readout fixture",
        "physical_device_validated": False,
        "channel_order": list(QUBITS),
        "readout": readout,
        "basis_dimension": model.basis.dimension,
        "receiver": receiver,
        "conditional_iq_centroids": {
            state: {"real": row.real.tolist(), "imag": row.imag.tolist()}
            for state, row in zip(PREPARED_STATES, centroids, strict=True)
        },
        "iq_distance_from_ground": dict(zip(PREPARED_STATES[1:], distances.tolist(), strict=True)),
        "zero_chi_max_iq_distance_from_ground": null_distance,
        "prepared_states": prepared,
        "prepared_state_addressed_report_probability_changes": prepared_readout_changes,
        "record_replay": replay,
        "microwave_pulse_control": {
            "pulse": pulse["drives"][0],
            "pulsed": pulsed,
            "matched_idle": matched_idle,
            "q0_level_one_probability_change": excitation_change,
            "q0_reported_one_probability_change": readout_change,
        },
        "checks": checks,
    }
    atomic_json(output / "pulse-to-iq.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Fresh directory for the report and run evidence")
    args = parser.parse_args(argv)
    report = run_example(args.output)
    print(json.dumps({"status": report["status"], "report": str(args.output / "pulse-to-iq.json"),
                      "checks": report["checks"]}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
