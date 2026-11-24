"""Reduce static applied SQUID flux from two independently meshed assemblies."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge, h

from ..io import atomic_json, digest, file_hash
from .assembly import FLUX_TABLES
from .design import ASSEMBLY_DESIGN_SCHEMA, DATA_QUBITS, MODE_ORDER
from .reduction import _case, _column, _field_provenance, _indices, _stage_output, _table

FLUX_TRANSFER_SCHEMA = "emulateqpu.fullwave.flux-transfer.v1"
STAGE = "magnetostatic"
COUPLERS = tuple(mode for mode in MODE_ORDER if mode not in DATA_QUBITS)
SOURCES = tuple(f"flux_{mode}" for mode in COUPLERS)
FLUX_QUANTUM_WB = h / (2 * elementary_charge)
PHI0_PER_MA_PER_H = 1e-3 / FLUX_QUANTUM_WB
MESH_RELATIVE_TOLERANCE = .05
MESH_ABSOLUTE_TOLERANCE_PHI0_PER_MA = 1e-4


def _setup(design):
    if design.get("schema") != ASSEMBLY_DESIGN_SCHEMA or "flux_transfer" not in design:
        raise ValueError("Static flux transfer requires an assembled design with magnetostatic inputs")
    # _case has verified the complete magnetostatic setup against this record.
    return design["flux_transfer"]


def _ordered_rows(table, wanted):
    result = {value: index for index, value in enumerate(_indices(table, "i"))}
    if set(result) != set(wanted):
        raise ValueError("Native magnetostatic rows differ from the frozen current source indices")
    return [result[index] for index in wanted]


def _flux_matrix(directory, setup):
    source_indices = [row["index"] for row in setup["sources"]]
    pickup_indices = [row["index"] for row in setup["pickups"]]
    surface, current = (_table(directory / name) for name in FLUX_TABLES)
    # Palace writes surface measurements in sorted source order using ordinal
    # rows. Its current table instead keeps the physical source indices.
    ordinal = {index: step for step, index in enumerate(sorted(source_indices), 1)}
    surface_row = _ordered_rows(surface, [ordinal[index] for index in source_indices])
    current_row = _ordered_rows(current, source_indices)
    excitation_A = _column(current, "I_inc[i] (A)")[current_row]
    if np.any(excitation_A <= 0):
        raise ValueError("Magnetostatic source currents must be positive in their declared orientation")
    pickup_Wb = np.stack([_column(surface, f"Φ_mag[{index}] (Wb)")[surface_row] for index in pickup_indices])
    with np.errstate(over="ignore", invalid="ignore"):
        transfer_H = pickup_Wb / excitation_A
    if not np.isfinite(transfer_H).all():
        raise ValueError("Nonfinite applied SQUID flux per source current")
    return transfer_H, excitation_A


def _mesh_agreement(coarse_H, fine_H):
    coarse = coarse_H * PHI0_PER_MA_PER_H
    fine = fine_H * PHI0_PER_MA_PER_H
    difference = np.abs(fine - coarse)
    limits = MESH_ABSOLUTE_TOLERANCE_PHI0_PER_MA + MESH_RELATIVE_TOLERANCE * np.maximum(np.abs(fine), np.abs(coarse))
    if not np.all(np.isfinite(difference)) or not np.all(np.isfinite(limits)):
        raise ValueError("Nonfinite flux-transfer mesh-agreement arithmetic")
    passed = difference <= limits
    return {"passed": bool(passed.all()), "entry_passed": passed.tolist(),
            "absolute_delta_Phi0_per_mA": difference.tolist(), "allowed_delta_Phi0_per_mA": limits.tolist(),
            "relative_tolerance": MESH_RELATIVE_TOLERANCE,
            "absolute_tolerance_Phi0_per_mA": MESH_ABSOLUTE_TOLERANCE_PHI0_PER_MA,
            "worst_delta_over_tolerance": float((difference / limits).max())}


def _read_verified_case(case, manifest, setup):
    directory = _stage_output(case, manifest, STAGE)
    files = _field_provenance(directory, FLUX_TABLES, solved_mesh=False, synthetic=False)
    transfer, current = _flux_matrix(directory, setup)
    if any(file_hash(directory / name) != sha for name, sha in files.items()):
        raise ValueError("Magnetostatic data changed during flux-transfer reduction")
    slurm = case / f"{STAGE}-slurm"
    receipt = slurm / "execution.json" if slurm.exists() else directory / "fullwave-execution.json"
    provenance = {"field_inputs_sha256": file_hash(case / "field-inputs.json"),
                  "mesh_sha256": manifest["files_sha256"]["mesh.msh"],
                  "config_sha256": manifest["files_sha256"][f"{STAGE}.json"],
                  "receipt_sha256": file_hash(receipt), "native_output_sha256": files,
                  "source_current_A": current.tolist()}
    return transfer, provenance


def reduce_flux_transfer(coarse_dir, fine_dir, output):
    """Write a hash-bound applied flux/current matrix and a two-mesh agreement report."""
    coarse_case, coarse_manifest, design = _case(coarse_dir, "coarse")
    fine_case, fine_manifest, fine_design = _case(fine_dir, "fine")
    coarse_setup = _setup(design)
    fine_setup = _setup(fine_design)
    coarse_input, fine_input = (manifest["files_sha256"] for manifest in (coarse_manifest, fine_manifest))
    if (digest(design) != digest(fine_design) or coarse_input["mesh.msh"] == fine_input["mesh.msh"]
            or coarse_input[f"{STAGE}.json"] != fine_input[f"{STAGE}.json"]):
        raise ValueError("Flux transfer requires distinct meshes of the same assembly and magnetostatic configuration")
    coarse, coarse_provenance = _read_verified_case(coarse_case, coarse_manifest, coarse_setup)
    fine, fine_provenance = _read_verified_case(fine_case, fine_manifest, fine_setup)
    check = _mesh_agreement(coarse, fine)
    report = {"schema": FLUX_TRANSFER_SCHEMA, "status": "MESH_AGREEMENT" if check["passed"] else "MESH_DISAGREEMENT",
              "passed": check["passed"], "design_id": design["design_id"], "design_sha256": digest(design),
              "readout_topology": design["readout_topology"], "loop_order": list(COUPLERS), "line_order": list(SOURCES),
              "flux_quantum_Wb": FLUX_QUANTUM_WB, "pickup_normal": "+Z", "source_and_pickup_geometry": fine_setup,
              "applied_flux_per_current_H": fine.tolist(),
              "applied_flux_per_current_Phi0_per_mA": (fine * PHI0_PER_MA_PER_H).tolist(),
              "coarse_applied_flux_per_current_H": coarse.tolist(), "mesh_agreement": check,
              "provenance": {"coarse": coarse_provenance, "fine": fine_provenance},
              "scope": "Static applied SQUID aperture flux derivative with the chosen passive PEC boundaries",
              "physical_device_calibrated": False, "dynamic_bias_transfer_modeled": False,
              "squid_loop_screening_modeled": False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "flux-transfer.json", report)
    return report
