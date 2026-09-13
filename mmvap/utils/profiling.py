"""
Lightweight opt-in step timer used to find bottlenecks in the eval pipeline.

Disabled by default (near-zero overhead: one env var check per call). Enable with:
    MMVAP_PROFILE=1 python ...
and call `mmvap.utils.profiling.summary()` (or just let scripts print it) once done.

Timings straddling a CUDA call are only meaningful with a `torch.cuda.synchronize()`
around them, since CUDA ops are launched asynchronously — otherwise a "fast" GPU
region just means the CPU raced ahead to the next `timed()` block. This module
synchronizes automatically inside `timed()` when CUDA is available, but that sync
itself is real wall-clock cost, so leave MMVAP_PROFILE unset for normal runs.
"""
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

ENABLED = os.environ.get("MMVAP_PROFILE", "0") == "1"

_timings = defaultdict(list)


@contextmanager
def timed(name: str):
    if not ENABLED:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _timings[name].append(time.perf_counter() - start)


def reset():
    _timings.clear()


def summary(top: int = None) -> str:
    if not _timings:
        return "Profiling summary: no timings recorded (MMVAP_PROFILE=1 not set?)"
    rows = sorted(_timings.items(), key=lambda kv: -sum(kv[1]))
    if top:
        rows = rows[:top]
    lines = [f"{'step':45s} {'count':>7s} {'total_s':>10s} {'mean_ms':>10s} {'%':>6s}"]
    grand_total = sum(sum(v) for v in _timings.values())
    for name, times in rows:
        total = sum(times)
        pct = 100 * total / grand_total if grand_total else 0.0
        lines.append(f"{name:45s} {len(times):7d} {total:10.2f} {1000*total/len(times):10.2f} {pct:5.1f}%")
    return "\n".join(lines)
