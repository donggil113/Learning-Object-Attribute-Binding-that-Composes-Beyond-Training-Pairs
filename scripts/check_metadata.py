#!/usr/bin/env python3
"""Build the scene-graph dataset, audit it, and run blind-detectability checks.

Outputs (in --out): log.txt, manifest.json, audit.json, blind.json, summary.json
and optionally data.jsonl. Decision rules are fixed here before running:
  * audit: every non-INFO check must PASS;
  * blind detectability (base vs. edited side from ONE modality): PASS if
    |AUC - 0.5| <= DETECT_TOL, else WARN (a WARN means the generator leaves a
    trace of which side was edited and must be fixed before any pilot).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bindcomp.dataset import audit, build_dataset, dataset_hash, dumps_jsonl  # noqa: E402
from bindcomp.encode import encode_groups  # noqa: E402
from bindcomp.evaluate import auc_bootstrap_ci  # noqa: E402
from bindcomp.manifest import RunRecorder, sha256_json, utc_stamp  # noqa: E402
from bindcomp.render import ProxyEncoder  # noqa: E402
from bindcomp.shortcuts import blind_detectability  # noqa: E402

DETECT_TOL = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_v0.json")
    ap.add_argument("--encoder-config", default="configs/encoder_v0.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-data", action="store_true")
    ap.add_argument("--blind-eval-splits", default="dev,calib,test_iid",
                    help="splits scored by the blind detectors (data_v1 keeps test splits unscored)")
    args = ap.parse_args()

    cfg = json.loads((ROOT / args.config).read_text())
    ecfg = json.loads((ROOT / args.encoder_config).read_text())
    out = Path(args.out) if args.out else ROOT / "runs" / f"metadata_check_{cfg['name']}_{utc_stamp()}"

    with RunRecorder("check_metadata", out) as rec:
        rec.extra["config_path"] = args.config
        rec.extra["config_sha256"] = sha256_json(cfg)
        rec.extra["encoder_config_sha256"] = sha256_json(ecfg)
        (out / "config.json").write_text(json.dumps({"data": cfg, "encoder": ecfg}, indent=2))

        rec.log(f"building dataset {cfg['name']}")
        ds, stats = build_dataset(cfg, log=rec.log)
        rec.extra["data_sha256"] = dataset_hash(ds)
        rec.extra["n_groups"] = {k: len(v) for k, v in ds.items()}
        rec.log(f"data sha256 {rec.extra['data_sha256']}")
        if args.save_data:
            (out / "data.jsonl").write_text(dumps_jsonl(ds))

        rep = audit(ds, cfg)
        rep["generation_stats"] = stats
        (out / "audit.json").write_text(json.dumps(rep, indent=2, default=str))
        for name, c in rep["checks"].items():
            if c["status"] != "INFO":
                rec.log(f"audit {c['status']:4s} {name} {json.dumps(c['detail'], default=str)}")
        rec.log(f"audit failures: {rep['n_fail']}")

        enc = ProxyEncoder(**{k: ecfg[k] for k in ("dim", "seed", "jitter", "lighting", "pos_scale")})
        train_enc = encode_groups(ds["train"], enc)
        eval_splits = args.blind_eval_splits.split(",")
        rec.extra["blind_eval_splits"] = eval_splits
        eval_enc = encode_groups([g for s in eval_splits for g in ds[s]], enc)
        blind = {}
        for modality in ("text", "image"):
            r = blind_detectability(train_enc, eval_enc, modality, seed=0)
            lo, hi = auc_bootstrap_ci(r["scores"], r["labels"], n_boot=200, seed=0)
            status = "PASS" if abs(r["auc"] - 0.5) <= DETECT_TOL else "WARN"
            blind[modality] = {"auc": r["auc"], "auc_ci95": [lo, hi], "n_train": r["n_train"],
                               "n_eval": r["n_eval"], "status": status, "tolerance": DETECT_TOL}
            rec.log(f"blind base-vs-edited detectability [{modality}] AUC={r['auc']:.4f} "
                    f"CI95=[{lo:.4f},{hi:.4f}] -> {status}")
        (out / "blind.json").write_text(json.dumps(blind, indent=2))

        ok = rep["n_fail"] == 0 and all(b["status"] == "PASS" for b in blind.values())
        summary = {
            "software_status": "TECHNICAL_TEST_PASS" if ok else "TECHNICAL_TEST_FAIL_OR_WARN",
            "science_status": "SCIENCE_NOT_EVALUATED",
            "audit_failures": rep["n_fail"],
            "blind": {k: v["status"] for k, v in blind.items()},
            "data_sha256": rec.extra["data_sha256"],
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        rec.extra["summary"] = summary
        rec.log(f"summary {json.dumps(summary)}")
    print(out)


if __name__ == "__main__":
    main()
