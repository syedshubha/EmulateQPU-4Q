# Local four-qubit electrostatic mesh screen

The same ideal chip was meshed at three resolutions on an Apple Silicon Mac
with 16 GiB of memory. Stock Palace v0.15.0 (`b676277`) ran all fourteen
ground-referenced terminal excitations on two Open MPI 5.0.9 ranks. The
coarsest mesh had 36,846 input tetrahedra. Palace bisected it to 36,876 during
boundary preprocessing. The medium and fine meshes had 59,999 and 121,184
tetrahedra, respectively.

The attached terminal-C tables are in farads and use the terminal order in
`results.json`. The study reduces them to a seven-mode differential
charging-energy matrix in GHz. Each adjacent comparison uses the finer result
as the denominator. The screen requires a relative Frobenius difference of
at most 2% for each matrix and a relative difference of at most 2% for each
charging-energy diagonal entry. Nearest-neighbor charging-energy changes are
also recorded in MHz.

First-order and second-order finite elements were compared on identical
tetrahedral meshes. The second-order run used full assembly. Both orders exceeded
the 2% threshold. On the medium-to-fine interval, the terminal-capacitance
and differential charging-energy differences fell from 13.02% and 10.98%
at first order to 5.59% and 5.29% at second order. The largest second-order
charging-energy diagonal change was 7.84% at `q1`. These tables support a local numerical diagnostic. Pinned
Palace v0.18.0 electrostatic qualification, electrodynamic mesh and port
qualification, and physical-device validation remain open.

To regenerate the inputs, use `python -m emulateqpu.fullwave.mesh_study prepare`
with `--order 1` for the first-order study or `--order 2 --full-assembly` for
the second-order study. The `mesh_study run` command accepts `--diagnostic` for
local development builds and `--mpi-ranks 2 --launcher mpirun` with a direct
Palace executable. Use `mesh_study analyze` to produce a fresh report. The
saved tables can also be checked without Palace by running the adjacent
`replay.py` script in the installed project environment with the `physical`
extra. The replay verifies the saved table hashes and recalculates the metrics.
