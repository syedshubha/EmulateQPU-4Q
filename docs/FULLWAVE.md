# Four-qubit field workflow

## Chip architecture

A dispersive readout measures the qubit through the state-dependent shift of a
coupled microwave resonator. Four resonators can share a frequency-multiplexed
line or use dedicated lines. Córcoles et al. demonstrated four independent
readout lines on a planar four-transmon chip. Spring et al. demonstrated
simultaneous readout of four qubits on a shared line with individual resonator
and Purcell-filter pairs. Both feedline topologies are available here on the
same four-qubit chip. [Córcoles et al., Methods and Figs. 6–7](https://www.nature.com/articles/ncomms7979) ·
[Spring et al., PRX Quantum 2025](https://doi.org/10.1103/PRXQuantum.6.020345) ·
[IBM and AWS hardware references](HARDWARE_REFERENCES.md)

The chain `q0–c01–q1–c12–q2–c23–q3` retains its seven island pairs and
linearized junctions. The four south-edge microwave drives couple to the data
qubits. Each data qubit also couples capacitively to an open-ended half-wave
coplanar waveguide (CPW) resonator. The two layouts use identical resonator
polygons and parallel coupling sections. The north edge carries eight readout
reference planes in the dedicated layout and two in the shared layout. Port
numbers 1–4 address `drive_q0` through `drive_q3` in both.

| Qubit | Dedicated input / output | Dedicated port numbers | Resonator length | Nominal frequency |
| --- | --- | --- | ---: | ---: |
| `q0` | `readout_q0_in` / `readout_q0_out` | 5 / 6 | 9.315 mm | 6.45 GHz |
| `q1` | `readout_q1_in` / `readout_q1_out` | 7 / 8 | 9.034 mm | 6.65 GHz |
| `q2` | `readout_q2_in` / `readout_q2_out` | 9 / 10 | 8.771 mm | 6.85 GHz |
| `q3` | `readout_q3_in` / `readout_q3_out` | 11 / 12 | 8.522 mm | 7.05 GHz |

The shared layout connects the four U-shaped sections along the north side
into a single feedline. Port 5 is `readout_in` near `q0` and port 6 is
`readout_out` near `q3`. Three ideal PEC ground straps cross the new
connections and bond the interior grounds to the enclosure ground. The straps
are 24 µm wide and rise to a 30 µm clearance above the feedline. The shared
route, ground pattern and straps all contribute to its electromagnetic response.

The readout CPW has a 10 µm center trace and a 6 µm ground gap outside its
coupling regions. The feedline and resonator run alongside one another for
400 µm with a 10 µm edge-to-edge gap. The nominal frequencies set the
half-wave centerline lengths using `ε_eff = (1 + ε_Si) / 2`. The pickup,
meander bends, feedline loading and enclosure shift actual resonances. The
silicon die is 7.7 mm by 6.05 mm and 0.35 mm thick in the model. Both layouts
use 50 Ω lumped terminations and fixed idle flux. The WKT polygons,
dimensions, bridge profiles, port orientations and readout mapping are
recorded in `design.json`.

To qualify the readout hardware, identify four resonator fields and fine-sweep
each through path to extract frequency, linewidth and external quality factor.
Examine field patterns for slotline and package modes, and use the full complex
scattering matrix to quantify reflections and isolation. For a state-dependent
twin, bind the dispersive shifts to a measurement chain with calibrated input
attenuation, output gain and noise, and digitizer demodulation. Fabrication
design still needs the coupler SQUID loops and flux routing, Purcell protection,
ground-bond and strap process details, PCB launches, and the fabrication process
stack. [Córcoles et al., cryogenic wiring in Fig. 6](https://www.nature.com/articles/ncomms7979) ·
[Heinsoo et al. on Purcell-filtered readout](https://doi.org/10.1103/PhysRevApplied.10.034040) ·
[Chen et al. on CPW ground airbridges](https://arxiv.org/abs/1310.2325)

## Prepare and solve

Install the `physical` extra on a host with Gmsh. Place new outputs on a scratch or shared results filesystem. `prepare` writes a new mesh, its physical labels, a configuration inventory, and four Palace inputs. The dedicated topology is the default. Add `--legacy` to prepare the original `q1`-only design.

```sh
uv sync --locked --extra physical
for topology in dedicated shared; do
  for refinement in coarse fine; do
    uv run --locked --extra physical python -m emulateqpu.fullwave prepare \
      "/scratch/qpu4/$topology-$refinement" --readout-topology "$topology" --refinement "$refinement"
  done
  uv run --locked --extra physical python -m emulateqpu.fullwave layout \
    "/scratch/qpu4/$topology-layout" --design "/scratch/qpu4/$topology-fine/design.json"
done
```

The GDSII export includes positive planar metal and named readout and junction
review markers. The shared layout includes three bridge projections on review
layer 103/0. The ideal bridge heights are in the frozen JSON and Gmsh geometry.
To export straight from CAD, use `layout OUTPUT --readout-topology shared`.

The full-wave configuration targets upstream Palace v0.18.0 at `b92aef83ecfe6d360c4b3d83e2122986297f6778`. Start with the upstream CPW and single-transmon examples using the same build that will run the chip. Freeze each reference mesh and configuration. For the CPW, compare complex scattering and power at a common reference plane with a port extension. For the single-transmon example, reproduce its eigenmode frequencies and junction participation before interpreting the four-qubit model. The upstream CPW example compares numeric wave ports and lumped ports. The chip prototype uses oriented lumped 50 Ω sheet ports with explicit reference planes. [Palace release](https://github.com/awslabs/palace/releases/tag/v0.18.0) · [CPW example sources](https://github.com/awslabs/palace/tree/b92aef83ecfe6d360c4b3d83e2122986297f6778/examples/cpw) · [Single-transmon example](https://awslabs.github.io/palace/stable/examples/transmon/)

Run both eigenmode inputs, then the fine electrostatic input and the two fine driven grids after allocating resources. Preserve the raw `palace.json` metadata, eigenmode tables, solved mesh parts, and electric fields. For local runs, use the bound launcher for each stage, for example `python -m emulateqpu.fullwave solve /scratch/qpu4/coarse eigenmode --executable /opt/palace/bin/palace`. On Slurm, plan each mesh/config pair and render it to the corresponding `*-slurm` directory under its case directory. The reducer consumes completed raw output and checks its run receipt, planned config hash, and input mesh hash. The electrostatic input uses the fine mesh, excites the fourteen islands, and grounds the auxiliary conductors. This is the explicit terminated static baseline for the same fixed junction biases. [HPC workflow](HPC.md)

The driven training and held-out scans are distinct input files. Each physical port receives its own excitation index. The default scan covers 1.8 to 10 GHz. The qualification report checks that all seven junctions have distinct participating modes inside that band. It also checks held-out complex scattering error across the complete matrix and in both directions of each physical readout path. Use a fine local sweep around any narrow feature, since resonances between training and held-out samples can evade these checks. To prepare a refined grid in a fresh directory, pass `--frequency-grids` with JSON `train_GHz` and `holdout_GHz` arrays.

For a three-level electrostatic ladder with fixed geometry, use the [mesh qualification workflow](MESH_QUALIFICATION.md).

## Adjust the layout and compare isolation

The bare-die `layout`, `prepare`, and `mesh_study prepare` commands accept `--drive-gap-um` to move all four XY pickups relative to their islands and `--readout-frequencies-ghz` to set the nominal resonator lengths in `q0` through `q3` order. Select `--readout-topology dedicated` or `shared`. The generated design records the polygons and has a content hash that follows it through GDS export, field preparation, and reduction. Resonator targets set the geometry. Solved resonances depend on loading, packaging, and the fields.

```sh
work=$(mktemp -d /tmp/qpu4-isolation.XXXXXX)
for topology in dedicated shared; do
  for mesh in coarse fine; do
    uv run --locked --extra physical python -m emulateqpu.fullwave prepare \
      "$work/$topology-$mesh" --refinement "$mesh" --readout-topology "$topology" \
      --drive-gap-um 40 --readout-frequencies-ghz 6.40 6.62 6.84 7.08
  done
  uv run --locked --extra physical python -m emulateqpu.fullwave layout \
    "$work/$topology-layout" --design "$work/$topology-fine/design.json"
done
```

Solve and qualify each candidate with the same idle biases, terminations, and sampling plan. Change one geometry setting at a time when measuring its effect. After `qualify` produces passing reductions, compare their small-signal RF isolation over the same frequency band.

```sh
uv run --locked python -m emulateqpu.fullwave compare-isolation \
  "$work/dedicated-reduction" "$work/shared-reduction" --band-ghz 4.5 6.0 \
  > "$work/isolation.json"
```

For each driven XY port, the report gives power fractions at the other drives and at the physically connected readout endpoints, along with source reflection. It uses the common solved frequency samples and matched 50 Ω reference planes. Choose the comparison band and sample density around the intended workload and inspect the reported sample grid. Use field-identified resonator populations, realized spacing and linewidths, and state-dependent receiver checks to judge whether a candidate also meets its control and readout requirements.

## Set up a matched readout crosstalk comparison

Qualify the coarse and fine meshes of each topology as a pair. Use the same
incident-power convention, idle biases, and initial scan grid. Identify the
four readout modes by their fields and sweep densely around the extracted
resonances. Record the realized resonance frequencies, linewidths, and
reference-plane scattering before comparing drive responses.

For the dedicated layout, driving `readout_qi_in` and observing
`readout_qj_out` measures port leakage for $j \ne i$. The shared
`readout_out` signal carries all four frequency channels. To compare the
layouts using the same quantity, select each target resonator in turn and
compare spectator-to-target mode population at the end of an equal-duration
weak tone. Use the extracted mode assignments and report any differences in
target loading and linewidth. Shared-output tone leakage also depends on the
demodulation chain, and assignment errors require a state-dependent model.

## Reduce verified fields

For single-rank saved eigenmodes of a planar silicon-and-air model, build the native MFEM overlap helper against the same MFEM and Palace source used for the fields. Its wrapper runs the analytic two-mesh fixture first, then compares two common-point quadrature rules on the actual saved electric fields. Missing physical points, material mismatches, and quadrature instability fail the overlap stage.

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

For mounted assemblies and distributed saves, supply samples from a separate native evaluator to `fullwave overlap`. The evaluator needs the assembly material permittivities and must account for every saved partition. The NumPy archive contains `points_mm` with shape `(points, 3)`, positive volume and permittivity `weights`, complex `coarse_E` and `fine_E` with shape `(modes, points, 3)`, and `coarse_modes` and `fine_modes` in `eig.csv` order. The scalar strings `coarse_fields_sha256` and `fine_fields_sha256` hold `emulateqpu.io.digest` of each eigenmode file-hash map. The map contains `palace.json`, `eig.csv`, `port-EPR.csv`, `port-Q.csv`, `domain-E.csv` and every solved mesh and electric field part path relative to the field root. Record the sampling method and confirm that a denser common-point rule supports the overlap.

The reducer screens eigen residuals, cross-mesh frequency and mode continuity, electric/magnetic energy, modal loss balance, and passive reciprocal scattering. It then compares complex interpolation from the training grid against independent held-out field samples. The acceptance cutoffs are in the qualification report. An incomplete or failed case cannot produce a passing model. The model uses first-order quartic energy-participation quantization and a separate small-signal response calculation. Use `electrostatic-cell.json` for the electrostatic circuit comparison and the EPR artifact for the field-derived response. [Palace energy-participation conventions](https://awslabs.github.io/palace/stable/reference/#energy-participation-ratios) · [EPR quantization](https://doi.org/10.1038/s41534-021-00461-8)

## Exercise the reduced model

```sh
uv run --locked python -m emulateqpu.fullwave response /scratch/qpu4/dedicated-reduction readout_q2_in 6.85
uv run --locked python -m emulateqpu.fullwave response /scratch/qpu4/shared-reduction readout_in 6.85
```

The response command accepts every RF port as a source. It reports modal populations at the end of a weak square tone and power fractions for all reference planes. For the examples above, the paired outputs are `readout_q2_out` and `readout_out`, respectively. Legacy artifacts also use `readout_in` and `readout_out`. The Python `fock_energy_GHz` function evaluates zero-, one-, and two-excitation energies. The prototype assumes independent harmonic modes for the driven envelope and uses the EPR model for the energy spectrum. Full nonlinear control, state-dependent readout and calibrated device predictions require separate validation.

## Qualify predictions for the mounted assembly

Use `prepare-assembly` for the mounted die and carrier. Freeze the geometry,
material stack, junction biases, terminations, and 15-port inventory for each
topology. Define the operating band and tolerances for each observable before
solving. Account for numerical and measurement uncertainty when comparing
predictions with hardware. [Mounted die and carrier](REFERENCE_HARDWARE.md)

### Mesh and port convergence

1. Run the mounted model on independent meshes with smaller elements near
   junctions, CPW gaps, bridge and wire-bond contacts, and dielectric
   interfaces. Use a separate polynomial-order comparison on fixed geometry
   and a tighter linear-solver tolerance on a representative case to check
   which errors dominate. Record input and solved mesh counts, solver settings,
   and receipts. Compare terminal capacitance and charging energies, including
   the mutual terms that affect the intended observable. Track eigenfrequencies,
   linewidths, and junction and port participation across refinements. Match
   modes using complex electric-field overlap. Track near-degenerate groups as
   subspaces when the individual mode labels depend on the basis.
2. Record the port reference planes, ordered labels, 50 Ω impedances, voltage
   directions, and incident-power convention. Use one source per excitation
   and retain all reflections and transmissions, including the loaded unused
   carrier routes. Compare lumped-sheet and numeric-wave port results on a
   representative launch or CPW coupon. De-embed feedline and fixture effects
   to the same planes before comparing complex amplitude and phase. Palace's
   CPW example illustrates port-dependent reflection and sensitivity to
   element size, polynomial order, and sweep tolerance. [Palace CPW reference](https://github.com/awslabs/palace/blob/b92aef83ecfe6d360c4b3d83e2122986297f6778/docs/src/examples/cpw.md#L105-L116) ·
   [NIST cryogenic CPW calibration at 4 K](https://www.nist.gov/publications/cryogenic-chip-situ-s-parameter-calibration-using-superconducting-coplanar-waveguides)
3. Use common direct-solve frequencies on both meshes and compare the complex
   $S_{ij}$ entries there. Keep an independent frequency grid for testing the
   reduced response between training points. Add dense direct sweeps around
   poles and notches, then halve the frequency step to check fitted frequencies
   and linewidths. Check weak couplings entry by entry, with an absolute
   tolerance floor and a relative term tied to each entry's magnitude.
   The bundled reducer checks interpolation against held-out samples on the
   fine mesh. Record cross-mesh scattering agreement as separate evidence.

### Readout and package response

Identify four readout-like modes using electric-field localization, junction
participation, and the through-path response. Include their filter-like modes
and any package, slotline, or launch modes in the operating band. For each
readout, report the realized frequency, loaded linewidth, external coupling,
and separation from neighboring channels. Fit complex scattering with a
circle model for an isolated resonance or a coupled resonator and filter model
for overlapping poles. Check the fit against held-out frequencies and an
independent time-domain ring-down. Compare package modes with calibrated
multiport spectra and, where available, qubit spectroscopy sensitive to modes
with weak coupling to the launches. [Probst et al. on complex resonance fitting](https://arxiv.org/abs/1410.3365) ·
[Spring et al., Appendix E](https://arxiv.org/html/2409.04967#A5) ·
[Huang et al., §IV.4](https://arxiv.org/html/2012.01438#S4.SS4)

The default 100 MHz training grid and its 50 MHz-offset holdout grid provide a
broad survey. Choose the local grid from the narrowest pole of interest.
For an isolated pole, the loaded power linewidth is
$\Delta f=f_r/Q_L=\kappa/(2\pi)$, where energy decays as $e^{-\kappa t}$
and field amplitude decays as $e^{-\kappa t/2}$. Report which trace was fit.
The reference assembly uses PEC conductors and zero dielectric loss tangent.
Its simulated linewidth includes port loading. Estimating a physical internal
quality factor requires independently constrained loss parameters or
measurements. [Complex resonance fits and coupling conventions](https://arxiv.org/abs/1410.3365)

### Connect the RF model to the pulse engine

Use the junction participation of the converged modes to derive the idle
multilevel Hamiltonian, and test its transition frequencies, anharmonicities
and dispersive shifts against spectroscopy. To predict delivered drives,
retain the frequency-dependent transfer from each external port to each
relevant internal mode or junction. Port-to-port scattering characterizes
the external network. Internal field projections or junction-port impedance
provide the additional drive matrix elements. Include the independently
characterized control wiring and use its causal time response when applying
a waveform. A magnetostatic flux-transfer matrix supplies the DC limit of the
bias network. A measured step response constrains the time dependence.
[Minev et al., energy-participation quantization](https://doi.org/10.1038/s41534-021-00461-8) ·
[Solgun et al., multiport impedance and drive coupling](https://arxiv.org/abs/1712.08154) ·
[Rol et al., on-chip pulse characterization](https://arxiv.org/abs/1907.04818)

For repeated readout and physical reset, carry the resonator and filter state
across windows, or validate a reduced model with the same measured memory.
Include independently characterized relaxation, thermal excitation and
receiver noise over the workload's operating range. Reserve separate traces
for checking residual photons, conditional populations, and spectator
response as the delay between operations changes.
[McClure et al., resonator ring-down](https://doi.org/10.1103/PhysRevApplied.5.011001) ·
[Heinsoo et al., multiplexed readout](https://doi.org/10.1103/PhysRevApplied.10.034040)

### Held-out hardware checks

Use device spectroscopy and calibrated pulse and readout data to fit the
Hamiltonian, XY and flux transfer, dissipation, and receiver model. Record the
chip-plane attenuation and phase, gain and noise, sample clock, integration
window, and classifier. Retain uncertainty estimates and identify the dataset
used for fitting. Predict separate measured traces and circuits before
scoring the following observables. [Energy-participation quantization and spectroscopy](https://arxiv.org/html/2010.00620) ·
[Heinsoo et al., selective and simultaneous readout](https://arxiv.org/html/1801.07904#S5)

| Workload | Held-out evidence |
| --- | --- |
| Reset | Measure post-reset populations from prepared `0`, `1`, and accessible leakage states, across delays and repeated cycles. Include matched idle and spectator measurements. |
| Pulses and gates | Compare driven population, phase, leakage, and spectator response for reserved waveforms and calibrated gate sequences. Use individual and simultaneous randomized benchmarking to test addressability under concurrent controls. [Gambetta et al.](https://arxiv.org/abs/1204.6308) |
| Mid-circuit measurement and feed-forward | Compare selective and simultaneous readout, repeated-readout transitions, conditional outcome distributions, spectator dephasing, and measured acquisition-to-action latency. Preserve raw IQ and classified records where available. |

Report the held-out residuals with their uncertainty, device configuration,
and calibration date. Scope hardware claims to the observable, workload, and
operating range covered by the results. Mesh and port convergence qualify the
linear field model. Device-specific predictions for reset, gates, and
mid-circuit measurement require the corresponding calibrated and held-out
measurements. [Magnard et al., reset and leakage](https://doi.org/10.1103/PhysRevLett.121.060502) ·
[Córcoles et al., hardware feedback experiments](https://arxiv.org/abs/2102.01682)
