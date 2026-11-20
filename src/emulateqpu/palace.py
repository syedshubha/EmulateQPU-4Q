"""Capability-detected Palace launch plans, separate from quantum propagation.

Detection runs help/version only. No field solve, native build, installation or
GPU inference occurs at import or planning time.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time

from ._resources import MemoryObservationError, sample_worker_rss_bytes
from .execution import _now, _read, _terminate, atomic_json

OFFICIAL_SOURCES = {
    "access_date": "2026-11-16",
    "running": "https://awslabs.github.io/palace/stable/run/",
    "parallelism": "https://awslabs.github.io/palace/stable/guide/parallelism/",
}


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def detect_palace(executable="palace", *, runner=None):
    """Read executable identity and bounded help/version output; never solve."""
    runner = runner or subprocess.run
    found = shutil.which(str(executable))
    if found is None:
        return {"available": False, "requested_executable": str(executable),
                "reason": "Executable not found", "documentation": OFFICIAL_SOURCES}
    path = Path(found).resolve()
    observations = {}
    for option in ("--help", "--version"):
        try:
            result = runner([str(path), option], capture_output=True, text=True, timeout=10, check=False)
            observations[option] = {"returncode": result.returncode,
                                    "stdout": result.stdout[:16000], "stderr": result.stderr[:4000]}
        except (OSError, subprocess.SubprocessError) as error:
            observations[option] = {"error": str(error)}
    help_text = observations["--help"].get("stdout", "") + observations["--help"].get("stderr", "")
    wrapper = "-np" in help_text and ("launcher" in help_text.lower() or "mpi" in help_text.lower())
    binaries = [{"path": str(p.resolve()), "sha256": _sha(p)}
                for p in sorted(path.parent.glob("palace-*.bin")) if p.is_file()]
    return {"available": True, "executable": str(path), "executable_sha256": _sha(path),
            "kind": "mpi_wrapper" if wrapper else "direct_binary_or_unknown",
            "wrapper_supports_threads": "-nt" in help_text,
            "wrapper_supports_launcher": "-launcher" in help_text,
            "sibling_binary_candidates": binaries, "observations": observations,
            "build_capabilities_verified": False, "gpu_support": "not established",
            "documentation": OFFICIAL_SOURCES}


def plan_palace(config, *, executable="palace", mpi_ranks=1, threads=1,
                launcher=None, launcher_args=None, build_manifest=None, runner=None,
                memory_mb=2048, wall_time_s=600):
    """Produce an exact command after checking declared build capabilities.

    build_manifest is an explicit supplied build record with optional
    executable_sha256, openmp, cuda and hip fields. The presence of a GPU does
    not establish a compatible Palace build.
    """
    for number, name, maximum in ((mpi_ranks, "mpi_ranks", 4096), (threads, "threads", 256),
                                   (memory_mb, "memory_mb", 1048576), (wall_time_s, "wall_time_s", 604800)):
        if type(number) is not int or not 1 <= number <= maximum:
            raise ValueError("Invalid " + name)
    config_path = Path(config).expanduser().resolve()
    cfg = _read(config_path)
    identity = detect_palace(executable, runner=runner)
    if not identity["available"]:
        raise ValueError(identity["reason"])
    manifest = _read(build_manifest) if isinstance(build_manifest, (str, Path)) else deepcopy(build_manifest or {})
    if manifest.get("executable_sha256", identity["executable_sha256"]) != identity["executable_sha256"]:
        raise ValueError("Build manifest belongs to a different executable")
    if threads > 1 and manifest.get("openmp") is not True:
        raise ValueError("Multiple threads require an explicit OpenMP build record")
    device = cfg.get("Solver", {}).get("Device", "CPU")
    if device == "GPU" and not (manifest.get("cuda") is True or manifest.get("hip") is True):
        raise ValueError("GPU solve requires an explicit CUDA/HIP build record")
    launcher_args = list(launcher_args or [])
    if any(not isinstance(x, str) or "\x00" in x or "\n" in x for x in launcher_args):
        raise ValueError("launcher_args must be literal command arguments")
    if any(int(os.environ.get(key, "1")) > 1 for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE")):
        raise ValueError("Nested MPI execution is unsupported; launch from the allocation's controlling shell")
    command = [identity["executable"]]
    if identity["kind"] == "mpi_wrapper":
        command += ["-np", str(mpi_ranks)]
        if identity["wrapper_supports_threads"]:
            command += ["-nt", str(threads)]
        elif threads > 1:
            raise ValueError("This Palace wrapper does not advertise thread configuration")
        if launcher is not None:
            if not identity["wrapper_supports_launcher"]:
                raise ValueError("This Palace wrapper does not advertise launcher selection")
            if launcher_args:
                raise ValueError("Wrapper launcher arguments require an explicit site wrapper; nesting is rejected")
            if shutil.which(launcher) is None:
                raise ValueError("MPI launcher not found")
            command += ["-launcher", launcher]
        elif launcher_args:
            raise ValueError("launcher_args require an explicit direct-binary launcher")
    else:
        if mpi_ranks > 1:
            if launcher not in ("mpirun", "mpiexec", "srun") or shutil.which(launcher) is None:
                raise ValueError("A detected mpirun/mpiexec/srun launcher is required for a direct binary")
            count = "--ntasks" if launcher == "srun" else "-n"
            command = [launcher, count, str(mpi_ranks), *launcher_args, *command]
        elif launcher is not None or launcher_args:
            raise ValueError("A direct serial binary uses no MPI launcher")
    command.append(str(config_path))
    return {"schema": "emulateqpu.palace_plan.v1", "created_utc": _now(),
            "config": str(config_path), "config_sha256": _sha(config_path),
            "command": command, "working_directory": str(config_path.parent),
            "executable_identity": identity, "supplied_build_manifest": manifest,
            "mpi_ranks": mpi_ranks, "threads_per_rank": threads, "device": device,
            "environment": {"OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "memory_mb": memory_mb, "wall_time_s": wall_time_s,
            "field_solve_executed": False, "physical_accuracy_qualified": False,
            "scope": "Palace field solver parallelism; independent of SciPy quantum tasks",
            "documentation": OFFICIAL_SOURCES}


def launch_palace(plan, output):
    """Explicit field execution; callers must separately authorize any solve.

    The config must already direct Palace output into this fresh directory.
    The adapter never rewrites a historical config or changes its mesh paths.
    """
    plan = _read(plan) if isinstance(plan, (str, Path)) else deepcopy(plan)
    if plan.get("schema") != "emulateqpu.palace_plan.v1":
        raise ValueError("Unsupported Palace plan")
    config = Path(plan["config"])
    if _sha(config) != plan["config_sha256"]:
        raise ValueError("Palace input changed since planning")
    executable = plan["executable_identity"]["executable"]
    if _sha(executable) != plan["executable_identity"]["executable_sha256"]:
        raise ValueError("Palace executable changed since planning")
    root = Path(output).expanduser().resolve()
    cfg = _read(config)
    destination = cfg.get("Problem", {}).get("Output")
    if not destination:
        raise ValueError("Palace Problem.Output must identify this fresh output directory")
    configured = Path(destination)
    if not configured.is_absolute():
        configured = config.parent / configured
    if configured.resolve() != root:
        raise ValueError("Palace config output differs from the requested fresh directory")
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "launch_plan.json", plan)
    environment = {**os.environ, **plan["environment"]}
    started = time.monotonic()
    peak = 0
    reason = None
    with (root / "stdout.log").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(plan["command"], cwd=plan["working_directory"], env=environment,
                                   stdout=stdout, stderr=stderr, start_new_session=(os.name == "posix"))
        try:
            while process.poll() is None:
                observed = sample_worker_rss_bytes(process)
                if observed is None:
                    break
                peak = max(peak, observed)
                if peak > plan["memory_mb"] * 1024**2:
                    reason = "MEMORY_LIMIT"
                elif time.monotonic() - started > plan["wall_time_s"]:
                    reason = "WALL_TIME_LIMIT"
                if reason:
                    _terminate(process)
                    break
                time.sleep(0.1)
        except MemoryObservationError:
            reason = "MEMORY_OBSERVATION_FAILED"
            _terminate(process)
            raise
        except BaseException:
            reason = "INTERRUPTED"
            _terminate(process)
            raise
        finally:
            report = {"schema": "emulateqpu.palace_execution.v1", "ended_utc": _now(),
                      "command": plan["command"], "returncode": process.poll(),
                      "status": "COMPLETE" if process.poll() == 0 and not reason else "FAILED",
                      "resource_stop": reason, "elapsed_s": time.monotonic() - started,
                      "observed_peak_rss_bytes": peak, "plan": plan,
                      "field_numerical_qualification": "Requires independent solved-output checks",
                      "physical_accuracy_qualified": False}
            atomic_json(root / "execution.json", report)
    return report
