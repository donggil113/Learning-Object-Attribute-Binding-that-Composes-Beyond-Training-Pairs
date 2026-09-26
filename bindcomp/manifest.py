"""Run manifests: code/config/data/model hashes, environment, time and memory."""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": head or "NO_COMMIT",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_files": status.splitlines()[:50] if status else [],
    }


def env_info():
    optional = {m: importlib.util.find_spec(m) is not None for m in ("numpy", "torch", "pytest", "PIL")}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "optional_packages_present": optional,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }


class RunRecorder:
    """Context manager measuring wall/CPU time and peak RSS of this process."""

    def __init__(self, name, out_dir):
        self.name = name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "log.txt"
        self.extra = {}

    def log(self, msg):
        line = f"[{_dt.datetime.now(_dt.timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def __enter__(self):
        self.started = _dt.datetime.now(_dt.timezone.utc).isoformat()
        self.t0 = time.perf_counter()
        self.c0 = time.process_time()
        self.git = git_info()
        return self

    def __exit__(self, exc_type, exc, tb):
        manifest = {
            "name": self.name,
            "status": "COMPLETED" if exc_type is None else f"FAILED: {exc_type.__name__}: {exc}",
            "started_utc": self.started,
            "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "wall_time_s": round(time.perf_counter() - self.t0, 3),
            "cpu_time_s": round(time.process_time() - self.c0, 3),
            # Linux reports ru_maxrss in KiB.
            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "git": self.git,
            "env": env_info(),
            "argv": sys.argv,
            **self.extra,
        }
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        if exc_type is not None:
            self.log(f"run failed: {exc_type.__name__}: {exc}")
        return False


def utc_stamp():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
