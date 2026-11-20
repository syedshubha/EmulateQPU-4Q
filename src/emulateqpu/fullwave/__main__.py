"""Four-data native field inputs, reduction gates, and modal response."""
import argparse
import json
import subprocess
from pathlib import Path

from ..io import read_json
from .design import prepare_design
from .reduction import STAGE_TABLES, qualify_reduction, solve_local, write_mode_overlap, write_native_overlap
from .response import drive_response, load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Write a fresh geometry, mesh and Palace inputs")
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--refinement", choices=("coarse", "fine"), default="coarse")
    prepare.add_argument("--frequency-grids", type=Path, help="Optional JSON train_GHz and holdout_GHz arrays")
    solve = commands.add_parser("solve", help="Run and bind a local single-rank Palace stage")
    solve.add_argument("case", type=Path)
    solve.add_argument("stage", choices=tuple(STAGE_TABLES))
    solve.add_argument("--executable", required=True)
    solve.add_argument("--memory-mb", type=int, default=4096)
    solve.add_argument("--wall-time-s", type=int, default=600)
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
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            grids = read_json(args.frequency_grids) if args.frequency_grids else None
            result = prepare_design(args.output, refinement=args.refinement, frequency_grids=grids)
        elif args.command == "solve":
            result = solve_local(args.case, args.stage, executable=args.executable,
                                 memory_mb=args.memory_mb, wall_time_s=args.wall_time_s)
        elif args.command == "overlap":
            result = write_mode_overlap(args.coarse, args.fine, args.samples, args.output)
        elif args.command == "native-overlap":
            result = write_native_overlap(args.coarse, args.fine, args.sampler, args.output)
        elif args.command == "qualify":
            result = qualify_reduction(args.coarse, args.fine, args.overlap, args.output)
        else:
            result = drive_response(load_model(args.artifact), args.port, args.frequency_GHz,
                                    args.duration_ns, args.incident_photons_per_ns)
    except (OSError, ImportError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("passed", True) and result.get("status") != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
