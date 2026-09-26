#!/usr/bin/env python3
"""Run a command under the stage CPU budget and book its CPU time in the ledger.

    python3 scripts/ledger_run.py --name NAME -- CMD [ARGS...]

The child gets RLIMIT_CPU = remaining budget, RLIMIT_AS = --max-as-gib, and
single-thread math-library environment variables. Its measured CPU time is
appended to the ledger whatever the exit status (failures are kept).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bindcomp.budget import CpuBudget  # noqa: E402

THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ledger", default="runs/pixel_v1_cpu_ledger.json")
    ap.add_argument("--cap-s", type=float, default=3600.0)
    ap.add_argument("--max-as-gib", type=float, default=3.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    budget = CpuBudget(ROOT / a.ledger, a.cap_s)
    rem = budget.remaining()
    if rem <= 5:
        print(f"NOT_RUN: CPU budget exhausted ({rem:.1f}s left)")
        sys.exit(3)
    limit_as = int(a.max_as_gib * 2**30)

    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (int(rem), int(rem) + 2))
        resource.setrlimit(resource.RLIMIT_AS, (limit_as, limit_as))

    env = dict(os.environ, **THREAD_ENV)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, env=env, preexec_fn=limits)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
    status = "COMPLETED" if p.returncode == 0 else f"FAILED(returncode={p.returncode})"
    entry = {"name": a.name, "cmd": cmd, "status": status, "cpu_s": round(cpu, 2),
             "wall_s": round(time.perf_counter() - t0, 2), "children_peak_rss_mib": round(r1.ru_maxrss / 1024, 1),
             "rlimit_cpu_s": int(rem), "rlimit_as_bytes": limit_as, "threads_env": THREAD_ENV}
    budget.append(entry)
    print(json.dumps({k: entry[k] for k in ("name", "status", "cpu_s", "wall_s", "children_peak_rss_mib")}),
          f"| ledger used {budget.used():.1f}/{a.cap_s:.0f}s")
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
