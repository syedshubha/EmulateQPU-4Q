"""Recalculate the mesh differences from the saved terminal-C tables."""
import math
from pathlib import Path

from emulateqpu.fullwave.design import four_qubit_design
from emulateqpu.fullwave.mesh_study import _interval, _matrices
from emulateqpu.io import digest, file_hash, read_json


def main():
    root = Path(__file__).resolve().parent
    saved = read_json(root / "results.json")
    design, _ = four_qubit_design(legacy=True)
    if digest(design) != saved["design_sha256"]:
        raise ValueError("The current chip design differs from the saved mesh study")

    for name, campaign in saved["campaigns"].items():
        matrices = {}
        for level, run in campaign["runs"].items():
            if run["status"] == "COMPLETE":
                table = root / run["capacitance_file"]
                if file_hash(table) != run["capacitance_sha256"]:
                    raise ValueError(f"Saved capacitance SHA-256 mismatch for {table.name}")
                matrices[level], _ = _matrices(table, design)

        for recorded in campaign["intervals"]:
            coarse, fine = recorded["from"], recorded["to"]
            interval = _interval(matrices[coarse], matrices[fine], design["mode_order"])
            if (interval["passed"] != recorded["passed"] or
                    any(not math.isclose(value, recorded["metrics"][key], rel_tol=1e-10, abs_tol=1e-12)
                        for key, value in interval["metrics"].items())):
                raise ValueError(f"Mesh differences changed for {name} {coarse} to {fine}")
            metrics = ", ".join(f"{key}={value:.2%}" for key, value in interval["metrics"].items())
            print(f"{name} {coarse} to {fine}  {metrics}")


if __name__ == "__main__":
    main()
