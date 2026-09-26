#!/usr/bin/env python3
"""Run the unittest suite and record a manifest (git hash, env, time, memory)."""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bindcomp.manifest import RunRecorder, utc_stamp  # noqa: E402


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / f"tests_{utc_stamp()}"
    with RunRecorder("unittest", out) as rec:
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT))
        buf = io.StringIO()
        result = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)
        (out / "unittest_output.txt").write_text(buf.getvalue())
        summary = {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "status": "TECHNICAL_TEST_PASS" if result.wasSuccessful() and not result.skipped else "TECHNICAL_TEST_FAIL",
        }
        rec.extra["summary"] = summary
        rec.log(json.dumps(summary))
    print(out)
    sys.exit(0 if summary["status"] == "TECHNICAL_TEST_PASS" else 1)


if __name__ == "__main__":
    main()
