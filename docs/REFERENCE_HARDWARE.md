# Four-qubit reference hardware

The reference layout uses a 7.7 × 6.05 mm die of 350 µm high-resistivity
silicon. The two versions use the same floating data transmons, tunable
couplers, readout resonators and Purcell filter geometry. Four separate
feedlines require 15 signal pads including the drives and flux lines. One
shared feedline requires nine. The published IBM and AWS measurements
inform the readout band and topology choices. [Hardware references](HARDWARE_REFERENCES.md)

## On-chip circuit

Four fixed-frequency data transmons each have one Al/AlOx/Al junction. Three
floating couplers each have two nominally matched junctions on parallel
branches. The oxide barrier is at the overlap of the two evaporated electrodes.
The aluminum bandage layer connects each electrode to its intended base
island outside the tunnel overlap. The nominal junction areas use 30 A/cm²
critical current density and 50 fF/µm² specific capacitance. Those values
set the initial dimensions and must be replaced by process data for
fabrication. Each DC SQUID has a dedicated grounded flux line beside its
aperture. The layout records the three current return paths and loop areas.
Signed flux per line current and the residual exchange and ZZ couplings
require extraction or measurement. [Floating couplers](https://arxiv.org/pdf/2103.07030) ·
[Aluminum junction and bandage process, Appendix F](https://arxiv.org/html/2509.04965#A6) ·
[Flux-line and ground-return measurements](https://doi.org/10.1063/5.0103597)

Each qubit loads the same half-wave readout resonator as in the field
prototype. In the reference layout, a half-wave Purcell filter couples each
resonator to its feedline. The readout resonator to filter and filter to
feedline gaps are both 10 µm. The closest direct feedline-to-readout
separation in the coupling region is 90 µm. Both layouts begin with nominal
readout and filter targets of 6.45, 6.65, 6.85 and 7.05 GHz. The coupled
resonances, bandwidth, and qubit-band rejection must be extracted from the
loaded network. Validate the filters by measuring both the readout
linewidths and the real admittance seen near the qubit frequencies.
[Individual Purcell filters for multiplexed readout](https://arxiv.org/pdf/1801.07904) ·
[Purcell filter analysis](https://arxiv.org/abs/1504.06030)

Each signal ends in a tapered 130 × 120 µm bond pad flanked by ground bond
sites. Signal sites are 145 µm in from the die edge. The nominal aluminum
ground bridges have 60 µm between landings and a 3 µm clearance above crossed
signals. Both layouts connect local coplanar grounds at feedline bends and
around the flux lines. The shared version adds three bridges at its north
joins. The chip checks that every coplanar ground region connects to a
package ground pad through the planar ground or a bridge.
[Released aluminum bridges](https://arxiv.org/html/1712.01671) ·
[Wire-bond and carrier design](https://arxiv.org/html/2012.01438#S3)

## Common carrier

Both dies fit a 9.4 × 9.7 mm alumina carrier. The 370 µm pocket accounts for
350 µm of silicon and a nominal 20 µm attach layer, with 75 µm clearance per
side. The die and carrier bond planes are coplanar. The carrier has 15
ground-signal-ground sites with estimated 50 Ω CPW sections. Its 80 µm
trace and 35.167 µm gap use the half-space alumina approximation.

The dedicated die populates all 15 sites. The shared die bonds its input and
output to the same first and last readout sites and leaves six carrier
routes unbonded at the die. The model specifies 50 Ω loads at the outer
reference planes of those six routes. Each active site has one nominal
25 µm aluminum signal wire and two ground wires with 100 µm loop height.
The carrier record contains the bond coordinates for each die and the
machining layer for the pocket. Connector transitions, chip-to-carrier
bond inductance and the enclosure lid require RF and mechanical validation.
[Package interconnect, bonds and cavity measurements](https://arxiv.org/html/2012.01438#S3)

## Export the layouts

```sh
uv sync --locked --extra physical
work=$(mktemp -d /tmp/qpu4-hardware.XXXXXX)
for topology in dedicated shared; do
  uv run --locked --extra physical python -m emulateqpu.fullwave layout \
    "$work/$topology" --hardware --readout-topology "$topology"
done
uv run --locked --extra physical python -m emulateqpu.fullwave carrier "$work/carrier"
```

The GDSII files use a 1 nm database grid. Each export has a frozen
`design.json` with the intended connectivity, assumptions, and polygons,
and a `layout.json` with the file hashes and layer map. The carrier and
the two dies have distinct top cell names.

| GDS layer/datatype | Geometry |
| --- | --- |
| 1/0 | Positive base aluminum, including the islands, ground, resonators, filters and launch traces |
| 2/0 and 3/0 | Intended bottom and top aluminum junction electrode footprints |
| 4/0 and 5/0 | Bridge contact openings and bridge aluminum |
| 6/0 | Aluminum junction-to-base bandages |
| 100/0–104/0 | Die edge, signal bond sites, tunnel overlaps, SQUID apertures and ground bond sites for review |

The junction layers describe final metal footprints. A fabrication process
must translate them to its lithographic profiles and verify alignment,
etching, oxide removal, contact resistance and bridge release.

## Prepare fields for the mounted die

`prepare-assembly` freezes the selected die and matching carrier, then
writes Gmsh geometry and Palace electrostatic, eigenmode and driven
inputs. This models the Purcell filters, recessed carrier pocket, wire bonds
and elevated ground bridges together. Use the same carrier and frequency
grids on both sides of a comparison.

```sh
for topology in dedicated shared; do
  for refinement in coarse fine; do
    uv run --locked --extra physical python -m emulateqpu.fullwave prepare-assembly \
      "$work/$topology-$refinement" --readout-topology "$topology" --refinement "$refinement" \
      --chip-layout "$work/$topology/design.json" --carrier-layout "$work/carrier/design.json"
  done
done
```

Omit both layout arguments to use the generated reference die and carrier.
Use a fresh output path for each case. The manifest binds the mesh,
configurations and frozen layout sources by SHA-256. Qualify coarse and fine
cases of each topology together using the [field workflow](FULLWAVE.md).

The two meshes expose the same fifteen carrier reference planes. Each
50 Ω port uses two lumped elements across the CPW gaps.
The shared die bonds its readout input to `readout_q0_in` and its readout
output to `readout_q3_out`. The six unbonded routes retain their 50 Ω
outer loads. Both driven configurations excite each carrier port separately
to support full scattering characterization. The model records the port
plane 90 µm in from the outer carrier edge.

Silicon and alumina are separate dielectric volumes. The 20 µm attachment
layer has an assumed relative permittivity of 3.2. A PEC lid and sidewall
enclose the assembly. The mesh uses zero-thickness PEC for the films and
bridges and 25 µm PEC ribbons for the three-point wire-bond profiles.
The ten junctions use 3 × 2 µm lumped sheets with the nominal branch
capacitance and inductance linearized at idle flux. Static extraction
uses the fourteen island terminals and adds the branch capacitances to
the reduced circuit. For eigenmode reduction, the two SQUID branches
remain separate in the quartic energy calculation. A fixed-bias model
omits dynamic flux transfer and cubic branch nonlinearities.
[Palace lumped port formulation](https://awslabs.github.io/palace/stable/reference/#lumped-ports)

Run the resulting cases through the same field workflow as the planar cases.
Local and Slurm receipts bind solver logs so the reducer can reconcile any
initial geometric subdivision with the solved mesh count. Resonance
locations, linewidths, crosstalk, flux transfer and package modes require
independent numerical or experimental qualification.

### Static flux transfer

Add `--flux-transfer` to both `prepare-assembly` commands for a given
topology. Use fresh case directories. This adds a magnetostatic input and
one passive pickup surface across each coupling SQUID's aperture. The windows
follow the aperture edges in the field mesh, including the two junction
sheets that close each aperture. The metal and enclosure use the assembled
PEC surfaces. Lumped ports are open, and the two oriented source sheets
for each flux line define the drive across its signal and return gaps.
[Palace boundary conditions](https://awslabs.github.io/palace/stable/guide/boundaries/)

For example, the following extracts the matrix for the shared-feedline
assembly. Set `palace` to the direct executable from the pinned Palace
v0.18.0 build used by the [field workflow](FULLWAVE.md).

```sh
for refinement in coarse fine; do
  uv run --locked --extra physical python -m emulateqpu.fullwave solve \
    "$work/shared-$refinement" magnetostatic --executable "$palace"
done
uv run --locked --extra physical python -m emulateqpu.fullwave reduce-flux \
  "$work/shared-coarse" "$work/shared-fine" "$work/shared-flux"
```

`flux-transfer.json` gives a 3 × 3 matrix in henries and Φ₀/mA. Rows follow
the SQUIDs `c01`, `c12`, and `c23` and columns follow their flux lines in
the same order. The positive pickup normal points above the die. Each
column divides the signed aperture flux by the current reported for the
corresponding source. The report records the coarse and fine solve receipts
and applies an absolute tolerance of 0.0001 Φ₀/mA plus 5% of the larger
magnitude in each entry. This check establishes agreement for those two
meshes under the selected boundary conditions. Physical device calibration,
SQUID loop screening, and the frequency response of the bias network are
outside this calculation.
[Palace magnetostatic source normalization](https://github.com/awslabs/palace/blob/b92aef83ecfe6d360c4b3d83e2122986297f6778/palace/drivers/magnetostaticsolver.cpp#L243-L337)
