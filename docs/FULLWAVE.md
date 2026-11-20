# Four-qubit field workflow

The device is a linear chain `q0–c01–q1–c12–q2–c23–q3` with four microwave drive sheets, seven oriented junction sheets, and one resonator coupled to `q1`. Two feedline terminations provide the readout input and output. The CAD uses the existing chain island and link geometry inside an ideal PEC package with a silicon substrate. All couplers use fixed idle flux during the comparison. Dimensions, linearized junction inductances, and port directions are recorded in `design.json`.

## Prepare and solve

Install the `physical` extra on a host with Gmsh. Place new outputs on a scratch or shared results filesystem. `prepare` writes a new mesh, its physical labels, a configuration inventory, and four Palace inputs. It performs no field solve.

```sh
uv sync --locked --extra physical
uv run --locked --extra physical python -m emulateqpu.fullwave prepare /scratch/qpu4/coarse --refinement coarse
uv run --locked --extra physical python -m emulateqpu.fullwave prepare /scratch/qpu4/fine --refinement fine
```

The full-wave configuration targets upstream Palace v0.18.0 at `b92aef83ecfe6d360c4b3d83e2122986297f6778`. Start with the upstream CPW and single-transmon examples using the same build that will run the chip. Freeze each reference mesh and configuration. For the CPW, compare complex scattering and power at a common reference plane with a port extension. For the single-transmon example, reproduce its eigenmode frequencies and junction participation before interpreting the four-qubit model. The upstream CPW uses numeric wave ports. The chip prototype instead uses oriented lumped 50-ohm sheet ports with explicit reference planes. [Palace release](https://github.com/awslabs/palace/releases/tag/v0.18.0) · [CPW example sources](https://github.com/awslabs/palace/tree/b92aef83ecfe6d360c4b3d83e2122986297f6778/examples/cpw) · [Single-transmon example](https://awslabs.github.io/palace/stable/examples/transmon/)

Run both eigenmode inputs, then the fine electrostatic input and the two fine driven grids after allocating resources. Preserve the raw `palace.json` metadata, eigenmode tables, solved mesh parts, and electric fields. For local runs, use the bound launcher for each stage, for example `python -m emulateqpu.fullwave solve /scratch/qpu4/coarse eigenmode --executable /opt/palace/bin/palace`. On Slurm, plan each mesh/config pair and render it to the corresponding `*-slurm` directory under its case directory. The reducer consumes completed raw output and checks its run receipt, planned config hash, and input mesh hash. The electrostatic input uses the fine mesh, excites the fourteen islands, and grounds the auxiliary conductors. This is the explicit terminated static baseline for the same fixed junction biases. [HPC workflow](HPC.md)

The driven training and held-out scans are distinct input files. Each physical port receives its own excitation index. The default scan covers 1.8 to 10 GHz. The qualification report checks that all seven junctions have distinct participating modes inside that band and can reject the grid if it misses a narrow resonant feature. To prepare a refined grid in a fresh directory, pass `--frequency-grids` with JSON `train_GHz` and `holdout_GHz` arrays.

For a three-level electrostatic ladder with fixed geometry, use the [mesh qualification workflow](MESH_QUALIFICATION.md).

## Reduce verified fields

For single-rank saved eigenmodes, build the native MFEM overlap helper against the same MFEM and Palace source used for the fields. Its wrapper runs the analytic two-mesh fixture first, then compares two common-point quadrature rules on the actual saved electric fields. Missing physical points, material mismatches, and quadrature instability fail the overlap stage.

```sh
cmake -S physical/fullwave-overlap -B /scratch/qpu4/overlap-build \
  -DPALACE_SOURCE=/path/to/palace -DMFEM_DIR=/path/to/mfem/cmake \
  -Dnlohmann_json_DIR=/path/to/nlohmann_json/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build /scratch/qpu4/overlap-build -j1
uv run --locked python -m emulateqpu.fullwave native-overlap /scratch/qpu4/coarse /scratch/qpu4/fine \
  /scratch/qpu4/overlap-build/emulateqpu_fullwave_overlap /scratch/qpu4/overlap
uv run --locked python -m emulateqpu.fullwave qualify /scratch/qpu4/coarse /scratch/qpu4/fine \
  /scratch/qpu4/overlap/overlap.json /scratch/qpu4/reduction
```

For distributed saves, `fullwave overlap` accepts samples from a rank-aware native evaluator. The NumPy archive contains `points_mm` with shape `(points, 3)`, positive volume and permittivity `weights`, complex `coarse_E` and `fine_E` with shape `(modes, points, 3)`, and `coarse_modes` and `fine_modes` in `eig.csv` order. The scalar strings `coarse_fields_sha256` and `fine_fields_sha256` hold `emulateqpu.io.digest` of each eigenmode file-hash map. The map contains `palace.json`, `eig.csv`, `port-EPR.csv`, `port-Q.csv`, `domain-E.csv` and every solved mesh and electric field part path relative to the field root. Record the sampling method and confirm that a denser common-point rule supports the overlap.

The reducer screens eigen residuals, cross-mesh frequency and mode continuity, electric/magnetic energy, modal loss balance, and passive reciprocal scattering. It then compares complex interpolation from the training grid against independent held-out field samples. The acceptance cutoffs are in the qualification report. An incomplete or failed case cannot produce a passing model. The model uses first-order quartic energy-participation quantization and a separate small-signal response calculation. Use `electrostatic-cell.json` for the electrostatic circuit comparison and the EPR artifact for the field-derived response. [Palace energy-participation conventions](https://awslabs.github.io/palace/stable/reference/#energy-participation-ratios) · [EPR quantization](https://doi.org/10.1038/s41534-021-00461-8)

## Exercise the reduced model

```sh
uv run --locked python -m emulateqpu.fullwave response /scratch/qpu4/reduction drive_q0 5.2
uv run --locked python -m emulateqpu.fullwave response /scratch/qpu4/reduction readout_in 6.8
```

The same response command accepts all four `drive_q*` ports. It reports modal populations at the end of a weak square tone and power fractions at the output reference planes, including `readout_out`. The Python `fock_energy_GHz` function evaluates zero-, one-, and two-excitation energies. The prototype assumes independent harmonic modes for the driven envelope and uses the EPR model for the energy spectrum. Full nonlinear control, state-dependent readout and calibrated device predictions require separate validation.
