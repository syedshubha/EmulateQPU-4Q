# Palace field jobs on Slurm

Use a shared filesystem visible to the submission and compute nodes. The field
profile names the direct Linux Palace executable, its build manifest, an
installed EmulateQPU-4Q Python environment, and the site's allocation. Use a build
manifest that records the executable's SHA-256 under `executable_sha256`. Keep
the manifest with the solver version and build information for the run.

```json
{
  "schema": "emulateqpu.field_slurm_profile.v1",
  "shared_filesystem": true,
  "shared_root": "/shared/qpu4",
  "python": "/shared/qpu4/venv/bin/python",
  "palace_binary": "/shared/qpu4/solver/palace-x86_64.bin",
  "build_manifest": "/shared/qpu4/solver/build.json",
  "partition": "cpu",
  "account": "example-account",
  "nodes": 1,
  "mpi_ranks": 2,
  "memory_mb": 2048,
  "wall_time": "00:05:00"
}
```

Set `mpi_type` only if the site requires an explicit Slurm MPI plugin. The
field workflow uses one `srun` step with one CPU thread per rank. The
allocation sets memory per node. The batch worker runs from the controlling
shell, using the scheduler's wall limit. Slurm accounts for remote MPI ranks.
The job step's `MaxRSS` is a maximum task value. [Slurm srun](https://slurm.schedmd.com/srun.html) · [Slurm sacct](https://slurm.schedmd.com/sacct.html)

Stage a CPU Palace `Electrostatic`, `Eigenmode`, or `Driven` case under the shared root. Planning
hashes the source config, mesh, direct executable, build record, and any
additional `--input` files. Additional paths resolve relative to the source
config. Rendering creates a fresh run directory, copies the config with an
absolute mesh path, and points `Problem.Output` into the run's `raw/` directory.

```sh
emulateqpu field plan case/eigenmode.json --profile field-site.json \
  --input field-inputs.json --require-table port-EPR.csv=2 --output field-plan.json
emulateqpu field render field-plan.json --output /shared/qpu4/case/eigenmode-slurm
emulateqpu field submit /shared/qpu4/case/eigenmode-slurm
emulateqpu field status /shared/qpu4/case/eigenmode-slurm
emulateqpu field collect /shared/qpu4/case/eigenmode-slurm
```

Set each `--require-table` count for the actual case. `terminal-C.csv` must have
at least one row and capacitance column per declared terminal. `eig.csv` must
have at least the configured mode count. `port-S.csv` is required for driven jobs. For
a driven sweep, also declare its expected frequency-row count, for example
`--require-table port-S.csv=21`. `status` combines verified outputs with
accounting for a successful job and one MPI step at the requested rank count.
Both `status` and `collect` also verify the solver's actual `Problem.MPISize`
from `raw/palace.json`. `collect` verifies output tables and hashes without
querying the scheduler. Use `raw/` at the original shared path for field-model
qualification, which includes mesh, port, and convergence checks.
For another attempt, confirm the previous job has stopped, render into a fresh
directory, and retain the previous records.

### Live acceptance pilot

1. With a site allocation, submit the same small eigenmode fixture twice on one
   node, changing only the rank count between one and two. Keep the same mesh,
   pinned solver build, and solver tolerances. Confirm both jobs through
   `field status` and `field collect`.
2. Compare frequencies of the same isolated modes across ranks. Use a fixed
   pilot tolerance of `1e-6 GHz` absolute and `1e-8` relative, and record the mode
   identities and residuals with the job receipts. Retain both elapsed times
   and the step accounting, including rank counts and available RSS.
