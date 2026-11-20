# Third-party dependencies

The Python distribution installs dependencies through their upstream
distributions. The native build uses a separately installed Palace-compatible
MFEM and nlohmann/json and the matching Palace source headers.

| Component | License | Role |
| --- | --- | --- |
| NumPy and SciPy | BSD-3-Clause | Arrays, numerical integration, and linear algebra |
| jsonschema | MIT | JSON contracts |
| threadpoolctl and psutil | BSD-3-Clause | Numerical threads and process resources |
| Shapely | BSD-3-Clause | Conductor polygons |
| Gmsh | GPL-2.0-or-later with upstream linking exception | CAD and meshing |
| Palace | Apache-2.0 | Native field solver |
| MFEM | BSD-3-Clause | Finite-element field recovery |
| nlohmann/json | MIT | Native overlap result serialization |
| QuTiP and scqubits | BSD-3-Clause | Independent numerical tests |

Installed dependencies may carry additional upstream terms and notices. Palace,
MFEM, the MPI runtime, and solver outputs are obtained or generated separately.
