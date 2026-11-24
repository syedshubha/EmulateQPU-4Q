# Slurm execution for Palace field jobs and circuit studies

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

Stage a CPU Palace `Electrostatic`, `Eigenmode`, `Driven`, or `Magnetostatic`
case under the shared root. Planning
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
`--require-table port-S.csv=21`. For current-sourced magnetostatic runs with
flux pickups, planning requires at least one row per source in both
`surface-F.csv` and `terminal-I.csv`. Render into the frozen case's
`magnetostatic-slurm` directory to use the
[static flux reducer](REFERENCE_HARDWARE.md#static-flux-transfer).
`status` combines verified outputs with
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

## Circuit study arrays

Use a circuit profile with the installed Python interpreter and a run directory
visible at the same path on all participating nodes. Set `max_array_size`
to the smaller of the site's `MaxArraySize` and any per-array task limit. The
default is 1001. Set `array_concurrency` to the number of independent circuit
workers the allocation permits. Allow time for Python startup, checkpointing,
and writing the result when setting the Slurm wall limit.
[Slurm array limits](https://slurm.schedmd.com/job_array.html)

```json
{
  "shared_filesystem": true,
  "shared_root": "/shared/qpu4",
  "python": "/shared/qpu4/venv/bin/python",
  "partition": "cpu",
  "memory_mb": 1024,
  "wall_time": "00:05:00",
  "array_concurrency": 2,
  "max_array_size": 1001,
  "max_retries": 1
}
```

```sh
emulateqpu hpc render study.json --profile circuit-site.json --output /shared/qpu4/circuit-run
emulateqpu hpc submit /shared/qpu4/circuit-run
emulateqpu hpc status /shared/qpu4/circuit-run
emulateqpu hpc collect /shared/qpu4/circuit-run
```

Large studies use successive arrays with local indices starting at zero and a
stable mapping to the original tasks. For an incomplete study,
`emulateqpu hpc submit /shared/qpu4/circuit-run --retry` checks queue and
accounting evidence before submitting the unfinished indices. After a worker
is killed, the retry archives its dispatch lock only if the request and task
match and Slurm confirms the exact owning cluster, array element, and job have
stopped. An indeterminate submission or a lock with ambiguous ownership
requires inspection. [Slurm job states](https://slurm.schedmd.com/job_state_codes.html)

A single-node Linux acceptance run with Slurm 24.11.5 and `MaxArraySize=2`
completed five four-qubit circuit tasks across three initial arrays. A retry
while tasks were active was refused. Cancelling one pending element led to a
retry of just that global task, and all five collected states and result
summaries matched serial execution exactly. In a separate run a running
element was killed. The retry archived its matching lock and completed the
task on the second attempt. These runs used a local container and Slurm
accounting. Production-site and multi-node acceptance remain open.

For site acceptance, run a small study that spans multiple arrays. Compare
collected circuit results with serial execution for the same task seeds.
Verify that an early retry is refused, cancel one pending test element, and
confirm that only its global task index is resubmitted. Retain the scripts,
task map, per-attempt receipts, and `sacct` records with the Slurm version,
node list, and site limits. Repeat on multiple allocated nodes before reporting
multi-node scaling. The Palace pilot above exercises MPI within a field solve.

Circuit arrays parallelize independent requests. Each worker reads the full
study and plan and evolves one request on one CPU process. Measure start-up
and total time as the study grows. For larger circuits or repeated measurements,
report the retained basis dimension, excitation cutoff, branch count, peak
memory, and numerical agreement as the cutoff increases. For example, the
uncapped three-level model with four data qubits and three couplers has 2,187
basis states. Two ideal full-register measurements with a reported leakage
outcome have up to 289 outcome sequences, above the current 256-branch limit.
