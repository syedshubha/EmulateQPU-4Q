# Contributing

Use Python 3.11 or later, uv, and just. Install the physical dependencies before
testing geometry, meshing, or reduction changes.

```sh
uv sync --locked --extra physical
just lint
just test
just build
```

Use `just reference` to compare the numerical engine to scqubits and QuTiP.
Record the solver build and mesh sizes when reporting a field discrepancy.
Include reproducible commands and the first error for execution failures.
Store generated simulations and site profiles outside the checkout.
