"""
Per-stage memory profile: which part of a run owns the peak.

A background thread samples process RSS and, on MPS, the Metal driver and
PyTorch allocator figures every `interval` seconds. `mark(name)` closes the
current stage; the profile reports each stage's peak for every series. The
pipeline stage hooks (trellis3.memory.install_stage_hooks) call `mark()`
automatically when a Profiler is active, so a normal `generate.py
--profile-memory` run yields the attribution without any other change.
"""

import threading
import time

from trellis3.memory import rss_gb


def _mps_readers():
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return {
                "mps_driver_gb": lambda: torch.mps.driver_allocated_memory() / 1e9,
                "mps_alloc_gb": lambda: torch.mps.current_allocated_memory() / 1e9,
            }
    except Exception:
        pass
    return {}


def _rss_now_gb():
    """Current RSS (not the max) in GB, or None."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        import os
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return None


class Profiler:
    def __init__(self, interval=0.05, readers=None):
        self.interval = interval
        self.readers = {"rss_gb": _rss_now_gb}
        self.readers.update(_mps_readers() if readers is None else readers)
        self.stages = []          # list of dicts
        self._cur = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.t0 = None

    # -- lifecycle -----------------------------------------------------------
    def start(self, first_stage="load"):
        self.t0 = time.time()
        self._open(first_stage)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._close()
        return self.report()

    # -- stages ---------------------------------------------------------------
    def mark(self, name):
        """End the current stage and begin `name`."""
        with self._lock:
            self._close()
            self._open(name)

    def _open(self, name):
        self._cur = {"stage": name, "start_s": round(time.time() - self.t0, 2),
                     "peak": {k: None for k in self.readers}, "samples": 0}
        self._sample()

    def _close(self):
        if self._cur is None:
            return
        self._sample()
        self._cur["seconds"] = round(time.time() - self.t0 - self._cur["start_s"], 2)
        self.stages.append(self._cur)
        self._cur = None

    def _sample(self):
        if self._cur is None:
            return
        for k, fn in self.readers.items():
            try:
                v = fn()
            except Exception:
                v = None
            if v is None:
                continue
            p = self._cur["peak"][k]
            self._cur["peak"][k] = v if p is None else max(p, v)
        self._cur["samples"] += 1

    def _loop(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                self._sample()

    # -- output ---------------------------------------------------------------
    def report(self):
        overall = {}
        for st in self.stages:
            for k, v in st["peak"].items():
                if v is not None:
                    overall[k] = max(overall.get(k, 0), v)
        max_rss = rss_gb()
        if max_rss:
            overall["max_rss_gb"] = max_rss
        return {"stages": self.stages, "overall_peak": overall}

    def table(self):
        keys = list(self.readers)
        head = f"{'stage':<28}{'s':>8}" + "".join(f"{k:>16}" for k in keys)
        rows = [head]
        for st in self.stages:
            row = f"{st['stage']:<28}{st['seconds']:>8.1f}"
            for k in keys:
                v = st["peak"].get(k)
                row += f"{'-':>16}" if v is None else f"{v:>16.2f}"
            rows.append(row)
        return "\n".join(rows)


_active = None


def activate(profiler):
    global _active
    _active = profiler


def mark(name):
    if _active is not None:
        _active.mark(name)
