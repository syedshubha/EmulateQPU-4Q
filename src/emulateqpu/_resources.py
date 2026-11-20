"""Process resource observations shared by the supported runtimes."""
from __future__ import annotations

import os
from subprocess import TimeoutExpired
import sys

import psutil


class MemoryObservationError(RuntimeError):
    """The operating system did not provide an expected memory reading."""


class WallTimeLimitError(TimeoutError):
    """The worker reached its cooperative wall deadline."""


def process_peak_rss_bytes():
    """Return this process's lifetime peak resident memory in bytes."""
    try:
        if os.name == "nt":
            peak = psutil.Process().memory_info().peak_wset
        else:
            import resource

            scale = 1 if sys.platform == "darwin" else 1024
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    except (OSError, psutil.Error) as error:
        raise MemoryObservationError(f"Cannot read process peak RSS: {error}") from error
    if peak <= 0:
        raise MemoryObservationError("Process peak RSS is unavailable")
    return int(peak)


def process_tree_rss_bytes(pid):
    """Return the sampled RSS of a process and its live descendants."""
    process = psutil.Process(pid)
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            # Descendants may exit between enumeration and observation.
            continue
    return rss


def sample_worker_rss_bytes(child_process):
    """Sample worker-tree RSS, returning None when the child has exited."""
    for exit_wait_s in (0.05, 0.2):
        try:
            rss = process_tree_rss_bytes(child_process.pid)
        except psutil.NoSuchProcess as error:
            observation_error = error
        except (OSError, psutil.Error) as error:
            raise MemoryObservationError(f"Cannot sample worker RSS: {error}") from error
        else:
            observation_error = None
            if child_process.poll() is not None:
                return None
            if rss > 0:
                return rss
        # Process metadata can be absent around spawn and exit. Retry once
        # and allow a bounded interval for the worker to be reaped.
        try:
            child_process.wait(timeout=exit_wait_s)
        except TimeoutExpired:
            continue
        return None
    if observation_error is not None:
        raise MemoryObservationError(f"Cannot sample worker RSS: {observation_error}") from observation_error
    raise MemoryObservationError("Worker RSS is unavailable")
