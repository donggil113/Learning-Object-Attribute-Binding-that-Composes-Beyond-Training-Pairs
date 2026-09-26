#!/usr/bin/env python3
"""Download the single pinned encoder checkpoint (stdlib only) and verify it.

Refuses to download more than ``max_download_bytes`` and deletes the file if the
size or sha256 does not match the pinned values. Records bytes, time and hash
in runs/fetch_encoder_<stamp>/manifest.json (download cost is reported
separately from the experiment CPU cap).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bindcomp.manifest import RunRecorder, sha256_file, utc_stamp  # noqa: E402


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/encoder_pixel_v1.json"
    cfg = json.loads((ROOT / cfg_path).read_text())
    dest = ROOT / cfg["local_path"]
    out = ROOT / "runs" / f"fetch_encoder_{utc_stamp()}"
    with RunRecorder("fetch_encoder", out) as rec:
        rec.extra["encoder_config"] = cfg
        if cfg["size_bytes"] > cfg["max_download_bytes"]:
            raise SystemExit("checkpoint exceeds the approved download size")
        if dest.exists() and sha256_file(dest) == cfg["sha256"]:
            rec.log(f"already present and verified: {dest}")
            rec.extra["downloaded_bytes"] = 0
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{cfg['hf_repo']}/resolve/{cfg['revision']}/{cfg['file']}"
        rec.log(f"GET {url}")
        tmp = dest.with_suffix(".part")
        h = hashlib.sha256()
        n = 0
        t0 = time.perf_counter()
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                n += len(chunk)
                if n > cfg["max_download_bytes"]:
                    f.close()
                    tmp.unlink()
                    raise SystemExit("download exceeded the approved size; aborted")
                h.update(chunk)
                f.write(chunk)
        digest = h.hexdigest()
        rec.extra.update(downloaded_bytes=n, download_wall_s=round(time.perf_counter() - t0, 1), sha256=digest)
        if n != cfg["size_bytes"] or digest != cfg["sha256"]:
            tmp.unlink()
            raise SystemExit(f"verification failed: size={n} sha256={digest}")
        tmp.rename(dest)
        rec.log(f"verified {n} bytes sha256={digest} -> {dest}")


if __name__ == "__main__":
    main()
