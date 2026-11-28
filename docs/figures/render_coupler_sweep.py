"""Sweep coupler c01 flux on the four-qubit example chain and plot frequencies and static ZZ.

The chain is the assumed fixture behind `emulateqpu init --example four-qubit`.
Each point rebuilds the dressed model with three levels per mode.

    uv run --locked --with matplotlib python docs/figures/render_coupler_sweep.py
    uv run --locked --with matplotlib python docs/figures/render_coupler_sweep.py --data docs/figures/coupler-sweep.json
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#152d47", "#405773", "#e3e8ef"
COLORS = {"q0": "#257de0", "q1": "#f47230", "c01": "#00b894", "zz01": "#6a4ce2", "zz12": "#666666"}


def sweep(points=45, max_flux=0.44):
    from emulateqpu.chain import linear_chain_cell
    from emulateqpu._engine.model import _local, build_model, static_zz_MHz

    base = linear_chain_cell(4)
    ic = base["mode_labels"].index("c01")
    ej_max = base["EJ_idle_GHz"][ic] / np.cos(np.pi * base["idle_flux_Phi0"][ic])
    rows = []
    for flux in np.linspace(0.0, max_flux, points):
        cell = copy.deepcopy(base)
        cell["idle_flux_Phi0"][ic] = float(flux)
        cell["EJ_idle_GHz"][ic] = float(ej_max * np.cos(np.pi * flux))
        model = build_model(cell, (3,) * len(cell["mode_labels"]))
        energies = model.encoded_energies
        zz = static_zz_MHz(model)
        coupler, _, _ = _local(base["EC_matrix_GHz"][ic][ic], cell["EJ_idle_GHz"][ic], 3, base["ncut"])
        rows.append({
            "flux_Phi0": round(float(flux), 6),
            "q0_GHz": float(energies[8] - energies[0]),   # |1000>
            "q1_GHz": float(energies[4] - energies[0]),   # |0100>
            "c01_bare_GHz": float(coupler[1]),
            "zz_q0_q1_kHz": zz["q0-q1"] * 1e3,
            "zz_q1_q2_kHz": zz["q1-q2"] * 1e3,
        })
    return rows


def chain_strip(ax):
    labels = ["q0", "c01", "q1", "c12", "q2", "c23", "q3"]
    xs = np.linspace(0.36, 0.97, len(labels))
    ax.plot([xs[0], xs[-1]], [0.5, 0.5], color=MUTED, lw=1.2, zorder=1)
    for x, label in zip(xs, labels):
        swept = label == "c01"
        qubit = label.startswith("q")
        ax.scatter([x], [0.5], s=900 if qubit else 650, marker="o" if qubit else "s", zorder=2,
                   facecolor=COLORS["c01"] if swept else "white", edgecolor=INK, linewidth=1.4)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9.5, color=INK,
                fontweight="bold" if swept else "normal", zorder=3)
    ax.text(xs[1], -0.25, "flux swept", ha="center", va="top", fontsize=8.5, color=MUTED)
    ax.text(0.0, 0.5, "Four transmons (circles) and\nthree tunable couplers (squares)", ha="left", va="center",
            fontsize=9, color=MUTED, linespacing=1.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")


def render(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "svg.fonttype": "path"})
    flux = np.array([r["flux_Phi0"] for r in rows])
    fig = plt.figure(figsize=(10, 5.2), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.55, 3], hspace=0.35, wspace=0.28,
                            left=0.07, right=0.97, top=0.9, bottom=0.14)
    chain_strip(fig.add_subplot(grid[0, :]))
    fig.suptitle("Tuning coupler c01 on the four-qubit example chain", x=0.07, ha="left",
                 fontsize=13, color=INK, fontweight="bold")

    ax = fig.add_subplot(grid[1, 0])
    for key, label, dy in (("q0_GHz", "q0", -5), ("q1_GHz", "q1", 5), ("c01_bare_GHz", "c01 (bare)", 0)):
        series = np.array([r[key] for r in rows])
        ax.plot(flux, series, color=COLORS[label.split()[0]], lw=2)
        ax.annotate(label, (flux[-1], series[-1]), xytext=(4, dy), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xlabel("Coupler flux, Φ/Φ₀")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_title("Mode frequencies", loc="left", fontsize=10.5, color=INK)

    bx = fig.add_subplot(grid[1, 1])
    zz01 = np.abs([r["zz_q0_q1_kHz"] for r in rows])
    zz12 = np.abs([r["zz_q1_q2_kHz"] for r in rows])
    bx.semilogy(flux, zz01, color=COLORS["zz01"], lw=2)
    bx.semilogy(flux, zz12, color=COLORS["zz12"], lw=2, ls=(0, (4, 3)))
    mark = int(np.searchsorted(flux, 0.12))
    bx.annotate("q0–q1 (through c01)", (flux[mark], zz01[mark]), xytext=(8, 4),
                textcoords="offset points", fontsize=9, color=INK)
    bx.annotate("q1–q2 (c12 fixed)", (flux[3], zz12[3]), xytext=(0, 6), textcoords="offset points",
                fontsize=9, color=INK)
    bx.set_xlabel("Coupler flux, Φ/Φ₀")
    bx.set_ylabel("|Static ZZ| (kHz)")
    bx.set_title("Residual ZZ between neighbors", loc="left", fontsize=10.5, color=INK)

    for a in (ax, bx):
        a.grid(True, color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.set_xlim(flux[0], flux[-1] * 1.18 if a is ax else flux[-1])
        a.set_xticks(np.arange(0, 0.41, 0.1))
    fig.text(0.07, 0.015, "Assumed circuit parameters (uncalibrated fixture); three levels per mode, dressed energies.",
             fontsize=8.5, color=MUTED)
    fig.savefig(output, facecolor="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, help="reuse a saved sweep instead of recomputing")
    parser.add_argument("--output", type=Path, default=HERE / "coupler-sweep.svg")
    args = parser.parse_args()
    if args.data:
        rows = json.loads(args.data.read_text())
    else:
        rows = sweep()
        (HERE / "coupler-sweep.json").write_text(json.dumps(rows, indent=1) + "\n")
    render(rows, args.output)


if __name__ == "__main__":
    main()
