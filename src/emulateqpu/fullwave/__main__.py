"""Four-data native field inputs, reduction gates, and modal response."""
import argparse
import json
import subprocess
from pathlib import Path

from ..io import read_json
from .design import READOUT_TOPOLOGIES, add_planar_design_arguments, four_qubit_design, prepare_design
from .layout import export_layout
from .reduction import STAGE_TABLES, qualify_reduction, solve_local, write_mode_overlap, write_native_overlap
from .response import drive_response, load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Write a fresh geometry, mesh and Palace inputs")
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--refinement", choices=("coarse", "fine"), default="coarse")
    prepare.add_argument("--frequency-grids", type=Path, help="Optional JSON train_GHz and holdout_GHz arrays")
    prepare.add_argument("--legacy", action="store_true", help="Prepare the original q1-only readout design")
    prepare.add_argument("--readout-topology", choices=READOUT_TOPOLOGIES, default="dedicated",
                         help="Choose four dedicated feedlines or one shared feedline (default: dedicated)")
    add_planar_design_arguments(prepare)
    assembled = commands.add_parser("prepare-assembly", help="Mesh the reference die, carrier and wire bonds")
    assembled.add_argument("output", type=Path)
    assembled.add_argument("--refinement", choices=("coarse", "fine"), default="coarse")
    assembled.add_argument("--readout-topology", choices=READOUT_TOPOLOGIES, default="dedicated")
    assembled.add_argument("--frequency-grids", type=Path, help="Optional JSON train_GHz and holdout_GHz arrays")
    assembled.add_argument("--chip-layout", type=Path, help="Frozen reference die design.json")
    assembled.add_argument("--carrier-layout", type=Path, help="Matching carrier design.json")
    assembled.add_argument("--flux-transfer", action="store_true",
                           help="Add passive SQUID pickups and magnetostatic flux-transfer inputs")
    layout = commands.add_parser("layout", help="Export a reviewable GDSII layout and layer map")
    layout.add_argument("output", type=Path)
    source = layout.add_mutually_exclusive_group()
    source.add_argument("--design", type=Path, help="Export an existing design.json")
    source.add_argument("--readout-topology", choices=READOUT_TOPOLOGIES,
                        help="Generate a layout for the given feedline topology (default: dedicated)")
    layout.add_argument("--legacy", action="store_true", help="Export the original q1-only readout design")
    layout.add_argument("--hardware", action="store_true", help="Export the multilayer reference hardware layout")
    add_planar_design_arguments(layout)
    carrier = commands.add_parser("carrier", help="Export the carrier shared by the two reference dies")
    carrier.add_argument("output", type=Path)
    solve = commands.add_parser("solve", help="Run and bind a local Palace stage")
    solve.add_argument("case", type=Path)
    solve.add_argument("stage", choices=tuple(STAGE_TABLES))
    solve.add_argument("--executable", required=True)
    solve.add_argument("--mpi-ranks", type=int, default=1)
    solve.add_argument("--launcher", choices=("mpirun", "mpiexec", "srun"))
    solve.add_argument("--build-manifest", type=Path)
    solve.add_argument("--memory-mb", type=int, default=4096)
    solve.add_argument("--wall-time-s", type=int, default=600)
    flux = commands.add_parser("reduce-flux", help="Compare signed loop flux per current across two meshes")
    flux.add_argument("coarse", type=Path)
    flux.add_argument("fine", type=Path)
    flux.add_argument("output", type=Path)
    for name in ("overlap", "native-overlap", "qualify"):
        command = commands.add_parser(name)
        command.add_argument("coarse", type=Path)
        command.add_argument("fine", type=Path)
        command.add_argument({"overlap": "samples", "native-overlap": "sampler", "qualify": "overlap"}[name], type=Path)
        command.add_argument("output", type=Path)
    response = commands.add_parser("response", help="Evaluate weak drive in a qualified modal artifact")
    response.add_argument("artifact", type=Path)
    response.add_argument("port")
    response.add_argument("frequency_GHz", type=float)
    response.add_argument("--duration-ns", type=float, default=20.0)
    response.add_argument("--incident-photons-per-ns", type=float, default=1e-5)
    compare = commands.add_parser("compare-isolation", help="Compare RF isolation across numerically qualified layouts")
    compare.add_argument("artifacts", type=Path, nargs="+")
    compare.add_argument("--band-ghz", type=float, nargs=2, metavar=("MIN", "MAX"),
                         help="Inclusive comparison band at common solved frequencies")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            grids = read_json(args.frequency_grids) if args.frequency_grids else None
            result = prepare_design(args.output, refinement=args.refinement, frequency_grids=grids,
                                    legacy=args.legacy, readout_topology=args.readout_topology,
                                    readout_frequencies_GHz=args.readout_frequencies_ghz, drive_gap_um=args.drive_gap_um)
        elif args.command == "prepare-assembly":
            from .assembly import prepare_assembly

            grids = read_json(args.frequency_grids) if args.frequency_grids else None
            chip = read_json(args.chip_layout) if args.chip_layout else None
            carrier = read_json(args.carrier_layout) if args.carrier_layout else None
            result = prepare_assembly(args.output, refinement=args.refinement, frequency_grids=grids,
                                      readout_topology=args.readout_topology, chip=chip, carrier=carrier,
                                      flux_transfer=args.flux_transfer)
        elif args.command == "layout":
            if ((args.design or args.hardware) and
                    (args.readout_frequencies_ghz is not None or args.drive_gap_um is not None)):
                layout.error("Planar design controls require a generated bare-die layout")
            if args.design:
                if args.legacy or args.hardware:
                    layout.error("--design supplies its own geometry and excludes --legacy and --hardware")
                design = read_json(args.design)
            elif args.hardware:
                if args.legacy:
                    layout.error("--hardware requires a four-resonator topology")
                from .hardware import reference_hardware_design

                design = reference_hardware_design(readout_topology=args.readout_topology or "dedicated")
            else:
                design, _ = four_qubit_design(legacy=args.legacy,
                                              readout_topology=args.readout_topology or "dedicated",
                                              readout_frequencies_GHz=args.readout_frequencies_ghz,
                                              drive_gap_um=args.drive_gap_um)
            result = export_layout(args.output, design=design)
        elif args.command == "carrier":
            from .carrier_layout import reference_carrier_design
            from .hardware import reference_hardware_design

            dedicated = reference_hardware_design()
            shared = reference_hardware_design(readout_topology="shared")
            result = export_layout(args.output, design=reference_carrier_design(dedicated, shared))
        elif args.command == "solve":
            result = solve_local(args.case, args.stage, executable=args.executable,
                                 mpi_ranks=args.mpi_ranks, launcher=args.launcher, build_manifest=args.build_manifest,
                                 memory_mb=args.memory_mb, wall_time_s=args.wall_time_s)
        elif args.command == "reduce-flux":
            from .flux_transfer import reduce_flux_transfer

            result = reduce_flux_transfer(args.coarse, args.fine, args.output)
        elif args.command == "overlap":
            result = write_mode_overlap(args.coarse, args.fine, args.samples, args.output)
        elif args.command == "native-overlap":
            result = write_native_overlap(args.coarse, args.fine, args.sampler, args.output)
        elif args.command == "qualify":
            result = qualify_reduction(args.coarse, args.fine, args.overlap, args.output)
        elif args.command == "compare-isolation":
            from .isolation import compare_isolation

            result = compare_isolation(args.artifacts, band_GHz=args.band_ghz)
        else:
            result = drive_response(load_model(args.artifact), args.port, args.frequency_GHz,
                                    args.duration_ns, args.incident_photons_per_ns)
    except (OSError, ImportError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("passed", True) and result.get("status") != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
