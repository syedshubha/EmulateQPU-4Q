# Four-qubit electrostatic mesh qualification

The mesh study uses one four-qubit chip geometry. The PEC enclosure, silicon substrate, fourteen island terminals, grounded auxiliary electrodes, and junction capacitances are fixed across the three levels. Gmsh refines near the island edges, the junction sheets, the drive pads, and the readout conductors. Each level uses its edge target inside a 6 µm band and transitions toward its bulk target over 90 µm from the edges. The three meshes are generated separately. [Gmsh size field reference](https://gmsh.info/doc/texinfo/gmsh.html#Mesh-size-fields)

| Level | Edge target | Bulk target | Tetrahedra observed with Gmsh 4.15.2 |
| --- | ---: | ---: | ---: |
| `ladder-coarse` | 18 µm | 0.55 mm | 36,846 |
| `ladder-medium` | 12 µm | 0.44 mm | 59,999 |
| `ladder-fine` | 8 µm | 0.35 mm | 121,184 |

## Run on a workstation

Install the optional physical dependencies. Use a new directory on each attempt so the mesh files, native logs, and failed runs remain available. Preparation records the exact mesh, geometry, and config hashes and the acceptance criteria before any solver runs.

```sh
uv sync --locked --extra physical
work=$(mktemp -d /tmp/qpu4-mesh.XXXXXX)
uv run --locked --extra physical python -m emulateqpu.fullwave.mesh_study \
  prepare "$work/study" --full-assembly
```

Use the direct Palace v0.18.0 CPU executable built from commit `b92aef83ecfe6d360c4b3d83e2122986297f6778`. Supply the build record as JSON with `source_commit` and `executable_sha256` for that executable. Preserve the source checkout/build record alongside the report. The direct executable can use one rank or a local MPI launcher. The `--full-assembly` setting freezes `Solver.PartialAssemblyOrder=3` at P2. Full assembly can trade higher memory use for lower solve time. Omit the flag at preparation for Palace's default assembly. [Pinned Palace configuration schema](https://github.com/awslabs/palace/blob/b92aef83ecfe6d360c4b3d83e2122986297f6778/scripts/schema/config-schema.json#L345-L350)

```sh
for level in ladder-coarse ladder-medium ladder-fine; do
  uv run --locked --extra physical python -m emulateqpu.fullwave.mesh_study \
    run "$work/study" "$level" \
    --executable /opt/palace/bin/palace-arch.bin \
    --build-manifest /opt/palace/build.json \
    --mpi-ranks 2 --launcher mpirun --memory-mb 8192 --wall-time-s 600
done
uv run --locked --extra physical python -m emulateqpu.fullwave.mesh_study \
  analyze "$work/study" "$work/report"
```

The launcher retains stdout, stderr, elapsed time, observed peak memory, and native output per level. The analyzer verifies the recorded build, rank/thread settings, inputs, and outputs. It confirms all fourteen terminal solves and zero solution-adaptive iterations. Palace may bisect elements at under-resolved interior boundaries during geometric preprocessing. Any change in effective tetrahedron count must match the exact adjustment Palace reports in its log, and the effective counts must increase across the ladder. [Palace preprocessing source](https://github.com/awslabs/palace/blob/b92aef83ecfe6d360c4b3d83e2122986297f6778/palace/utils/geodata.cpp#L3222-L3236)

For laptop diagnostics, prepare the same meshes with `--order 1` or use another local Palace build. Pass `--diagnostic` to each `run` under those conditions. `register` can also freeze pre-generated ladder cases, and `analyze --diagnostic-unbound` can read an earlier standard local launcher receipt. Use the same `--order` and `--full-assembly` choices with `register` that were used for preparation. For Palace wrappers that split paths at spaces, use a scratch path without spaces or a direct executable. Diagnostic results carry a separate status and cannot certify the P2 pinned configuration.

## Interpret the report

The analyzer uses the 14 × 14 terminal capacitance matrix $C$ and evaluates the 7 × 7 differential charging-energy matrix $E_C/h$ in GHz using the fixed junction capacitances and a linear solve. It checks the terminal inventory, finite entries, reciprocity, the Maxwell capacitance sign convention, and positive definiteness. It screens both successive mesh intervals with the same registered limits.

| Quantity | Limit for each interval |
| --- | ---: |
| Frobenius norm of the terminal $C$ change, divided by the refined matrix norm | 2% |
| Frobenius norm of the differential $E_C/h$ change, divided by the refined matrix norm | 2% |
| Largest relative change among the seven diagonal $E_C/h$ entries | 2% |

The report also retains each matrix, the diagonal changes by mode, and signed off-diagonal $E_C/h$ changes in MHz for consecutive qubit–coupler pairs. A failed or incomplete study writes `mesh-report.json` and exits with status `2`. Only a completed preregistered P2 study with verified source binding and both intervals below every limit sets `electrostatic_mesh_qualified` to `true`. Underlying modes, electromagnetic response, and measurements have separate acceptance criteria. [Electrodynamic workflow](FULLWAVE.md)

## Local diagnostic result

Gmsh 4.15.2 generated all three levels on an Apple M1. Palace v0.15.0 (`b676277`), at P1 with two MPI ranks, completed all fourteen terminal solves at each level. Palace added 30 tetrahedra to the coarse mesh during geometric preprocessing and reported zero solution-adaptive iterations on all levels. The diagnostic result was `DIAGNOSTIC_NOT_CONVERGED`.

| Interval | Terminal $C$ change | Differential $E_C/h$ change | Largest diagonal change | Largest nearest-neighbor change |
| --- | ---: | ---: | ---: | ---: |
| Coarse → medium | 20.16% | 17.03% | 20.84% | 1.541 MHz |
| Medium → fine | 13.02% | 10.98% | 16.33% | 0.958 MHz |

A full-assembly P2 run with the same local Palace build also completed. The
medium-to-fine changes were 5.59% for terminal capacitance, 5.29% for the
differential charging-energy matrix, and 7.84% for the largest diagonal
change. The three values exceeded the 2% limits. The laptop screen records
both solver configurations, mesh counts, and capacitance tables. [Mesh screen evidence](../reproducibility/mesh-ladder/README.md)
