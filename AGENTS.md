# Development instructions

Use `uv sync --locked --extra physical` for development. Run `just lint` and
`just test` before submitting changes. Run `just reference` after changing
the circuit solver. Keep generated meshes, solver outputs, and site profiles
outside the source tree. Preserve distinctions between circuit verification,
mesh qualification, and live scheduler tests in documentation and reports.
