"""Stage-wide CPU budget ledger and per-process caps (stdlib only).

Every stage run appends its measured CPU seconds (self + children) to one
ledger file; a new run may only start with the remaining budget as its hard
RLIMIT_CPU. The address-space limit is used to enforce the memory cap (it is
stricter than a peak-RSS cap because it also bounds unused mappings).
"""

from __future__ import annotations

import json
import os
import resource
import signal
import time
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


def _on_xcpu(signum, frame):
    raise BudgetExceeded("CPU budget exhausted (SIGXCPU)")


class CpuBudget:
    def __init__(self, ledger_path, cap_s):
        self.path = Path(ledger_path)
        self.cap_s = float(cap_s)

    def entries(self):
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text())["entries"]

    def used(self):
        return sum(e["cpu_s"] for e in self.entries())

    def remaining(self):
        return self.cap_s - self.used()

    def append(self, entry):
        entries = self.entries() + [entry]
        self.path.write_text(json.dumps({"cap_cpu_s": self.cap_s, "entries": entries,
                                         "used_cpu_s": round(sum(e["cpu_s"] for e in entries), 2)}, indent=2))

    def start(self, name, max_as_bytes=None, reserve_s=5.0):
        """Apply caps to this process; refuse to start without budget."""
        rem = self.remaining()
        if rem <= reserve_s:
            raise BudgetExceeded(f"no CPU budget left for {name}: remaining {rem:.1f}s")
        used_self = resource.getrusage(resource.RUSAGE_SELF)
        already = used_self.ru_utime + used_self.ru_stime
        soft = int(already + rem - reserve_s)
        resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + int(reserve_s)))
        signal.signal(signal.SIGXCPU, _on_xcpu)
        if max_as_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (int(max_as_bytes), int(max_as_bytes)))
        self._name = name
        self._t0 = time.perf_counter()
        self._c0 = already
        self._ch0 = self._children()
        return {"remaining_before_s": round(rem, 2), "rlimit_cpu_soft_s": soft, "rlimit_as_bytes": max_as_bytes,
                "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}}

    @staticmethod
    def _children():
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        return r.ru_utime + r.ru_stime

    def finish(self, status, run_dir, extra=None):
        r = resource.getrusage(resource.RUSAGE_SELF)
        cpu = (r.ru_utime + r.ru_stime) - self._c0 + (self._children() - self._ch0)
        entry = {"name": self._name, "run_dir": str(run_dir), "status": status, "cpu_s": round(cpu, 2),
                 "wall_s": round(time.perf_counter() - self._t0, 2),
                 "peak_rss_mib": round(r.ru_maxrss / 1024, 1), **(extra or {})}
        self.append(entry)
        return entry
