# Contributing

Use Python 3.11 or later, uv, and just. Install the physical dependencies before
testing geometry, meshing, or reduction changes.

```sh
uv sync --locked --extra physical
just lint
just test
just build
```

Run `just reference` after changing the circuit solver; it compares the numerical
engine to scqubits and QuTiP.
Record the solver build and mesh sizes when reporting a field discrepancy.
Include reproducible commands and the first error for execution failures.
Store generated meshes, solver outputs, and site profiles outside the checkout.
In documentation and reports, keep circuit verification, mesh qualification, and
live scheduler tests distinct.
