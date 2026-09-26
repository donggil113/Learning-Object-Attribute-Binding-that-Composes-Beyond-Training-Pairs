#!/usr/bin/env python3
"""End-to-end technical smoke run (NOT a pilot).

Trains the shared content/binding head under each loss variant for a few
steps on the train split, evaluates every scorer (trained heads, their
content-/binding-only channels, text-only, image-only, random, oracle) with the
same pair/group code, and writes collapse reports before and after training.

Outputs: log.txt, manifest.json, config.json, metrics.json, collapse.json,
history.json, heads/<variant>_seed<k>.json (parameters + hash).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bindcomp.collapse import head_report  # noqa: E402
from bindcomp.dataset import build_dataset, dataset_hash  # noqa: E402
from bindcomp.encode import encode_groups  # noqa: E402
from bindcomp.evaluate import full_report  # noqa: E402
from bindcomp.head import FactorizedHead  # noqa: E402
from bindcomp.manifest import RunRecorder, sha256_json, utc_stamp  # noqa: E402
from bindcomp.render import ProxyEncoder  # noqa: E402
from bindcomp.shortcuts import (HeadScorer, ImageOnlyScorer, OracleScorer, RandomScorer,  # noqa: E402
                                TextOnlyScorer)
from bindcomp.train import train  # noqa: E402


def compact(rep):
    """Drop per-kind detail for the log line."""
    return {k: {m: round(v, 3) for m, v in r["overall"].items() if m != "n"}
            for k, r in rep.items() if isinstance(r, dict) and "overall" in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke_v0.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scfg = json.loads((ROOT / args.config).read_text())
    dcfg = json.loads((ROOT / scfg["data_config"]).read_text())
    ecfg = json.loads((ROOT / scfg["encoder_config"]).read_text())
    out = Path(args.out) if args.out else ROOT / "runs" / f"{scfg['name']}_{utc_stamp()}"

    with RunRecorder("smoke_train", out) as rec:
        rec.extra.update(
            config_path=args.config,
            config_sha256=sha256_json(scfg),
            data_config_sha256=sha256_json(dcfg),
            encoder_config_sha256=sha256_json(ecfg),
            purpose=scfg["purpose"],
        )
        (out / "config.json").write_text(json.dumps({"smoke": scfg, "data": dcfg, "encoder": ecfg}, indent=2))

        ds, _ = build_dataset(dcfg)
        rec.extra["data_sha256"] = dataset_hash(ds)
        rec.log(f"data sha256 {rec.extra['data_sha256']}")
        enc = ProxyEncoder(**{k: ecfg[k] for k in ("dim", "seed", "jitter", "lighting", "pos_scale")})
        train_enc = encode_groups(ds["train"], enc)
        cap_n = scfg["eval_max_groups_per_split"]
        eval_enc = {s: encode_groups(ds[s][:cap_n], enc) for s in scfg["eval_splits"]}
        probe = eval_enc["dev"]

        metrics, collapse, history, timing = {}, {}, {}, {}

        def evaluate(name, scorer):
            t0 = time.perf_counter()
            rep = full_report(scorer, eval_enc, n_boot=scfg["n_boot"])
            metrics[name] = rep
            rec.log(f"eval {name} ({time.perf_counter() - t0:.1f}s): {json.dumps(compact(rep))}")

        evaluate("random", RandomScorer(seed=0))
        evaluate("oracle", OracleScorer())
        blind_train = train_enc[: scfg["blind_train_max_groups"]]
        evaluate("text_only", TextOnlyScorer().fit(blind_train))
        evaluate("image_only", ImageOnlyScorer().fit(blind_train))

        hcfg = scfg["head"]
        (out / "heads").mkdir(exist_ok=True)
        for seed in scfg["seeds"]:
            for variant in scfg["variants"]:
                tag = f"{variant}_seed{seed}"
                head = FactorizedHead(d=ecfg["dim"], seed=seed, **hcfg)
                rec.log(f"{tag}: n_params={head.n_params()} init_hash={head.param_hash()[:16]}")
                collapse[f"{tag}:init"] = head_report(head, probe)
                t0 = time.perf_counter()
                history[tag] = train(head, train_enc, scfg["train"], variant, seed, log=rec.log)
                timing[tag] = {"train_wall_s": round(time.perf_counter() - t0, 2),
                               "steps": scfg["train"]["steps"],
                               "groups_seen": scfg["train"]["steps"] * scfg["train"]["batch_groups"]}
                collapse[f"{tag}:final"] = rep = head_report(head, probe)
                rec.log(f"{tag}: collapse flags {rep['flags']} "
                        f"bind_sens img={rep['binding_sensitivity_img']:.3g} txt={rep['binding_sensitivity_txt']:.3g} "
                        f"content_residual_img={rep['content_residual_img']:.2e}")
                (out / "heads" / f"{tag}.json").write_text(json.dumps(
                    {"param_sha256": head.param_hash(), **head.state()}))
                timing[tag]["param_sha256"] = head.param_hash()
                for ch in ("all", "content", "binding"):
                    evaluate(f"{tag}[{ch}]", HeadScorer(head, ch))

        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        (out / "collapse.json").write_text(json.dumps(collapse, indent=1))
        (out / "history.json").write_text(json.dumps(history))
        rec.extra["timing"] = timing
        flags = {k: v["flags"] for k, v in collapse.items() if k.endswith(":final")}
        rec.extra["final_collapse_flags"] = flags
        rec.extra["software_status"] = "SMOKE_COMPLETED"
        rec.extra["science_status"] = "SCIENCE_NOT_EVALUATED"
        rec.log(f"final collapse flags: {json.dumps(flags)}")
    print(out)


if __name__ == "__main__":
    main()
