#!/usr/bin/env python3
"""Stage pixel_baseline_v1: pixel inputs + same-data hard-negative baseline (feasibility only).

Runs exactly what configs/pixel_baseline_v1.json fixes: data_v1 train/dev panels,
renderer integrity, frozen CLIP features from pixels/captions, controls, a fit
sanity run, the hardneg baseline for the fixed seeds, gradient ratios on train
batches, diagnostics and the pre-declared decision rules. The whole process
runs under the stage CPU ledger (RLIMIT_CPU), one torch thread and a 3 GiB
address-space cap.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import mean, median  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from bindcomp import pixel_render  # noqa: E402
from bindcomp.budget import BudgetExceeded, CpuBudget  # noqa: E402
from bindcomp.clip_features import ClipEncoder  # noqa: E402
from bindcomp.dataset import build_dataset, dataset_hash  # noqa: E402
from bindcomp.encode import encode_groups  # noqa: E402
from bindcomp.evaluate import (auc, auc_bootstrap_ci, binding_necessary, bootstrap_ci, evaluate_groups,  # noqa: E402
                               pair_scores, summarize, tie_rate)
from bindcomp.manifest import RunRecorder, sha256_file, sha256_json, utc_stamp  # noqa: E402
from bindcomp.pixel import (ClipImageOnlyScorer, FeatGroup, ZeroShotClipScorer, apply_standardizer,  # noqa: E402
                            base_vs_edited_detectability, build_feature_groups, cosine_distance,
                            render_members, shuffled, standardizer)
from bindcomp.render import ProxyEncoder  # noqa: E402
from bindcomp.shortcuts import OracleScorer, RandomScorer, TextOnlyScorer  # noqa: E402
from bindcomp.torch_head import (DTYPE, TorchFactorizedHead, TorchHeadScorer, batch_loss,  # noqa: E402
                                 grad_norm_of, train)

CHANCE = {"text": 0.25, "image": 0.25, "group": 1 / 6, "match": 0.5, "aug": 1 / 3}


def report(scorer, fgs, n_boot=500):
    rows = evaluate_groups(scorer, fgs)
    bn_rows = [r for r in rows if binding_necessary(r["kind"])]
    bn_fgs = [fg for fg in fgs if binding_necessary(fg.kind_label)]
    s_all, l_all = pair_scores(rows, fgs)
    s_bn, l_bn = pair_scores(bn_rows, bn_fgs)
    out = {
        "provenance": getattr(scorer, "provenance", "n/a"),
        "binding_necessary": summarize(bn_rows).get("all"),
        "all": summarize(rows)["all"],
        "by_kind": summarize(rows, key=lambda r: r["kind"]),
        "binding_necessary_group_ci95": bootstrap_ci([r["group"] for r in bn_rows], n_boot=n_boot) if bn_rows else None,
        "pair_auc_all": auc(s_all, l_all),
        "pair_auc_binding_necessary": auc(s_bn, l_bn) if bn_rows else None,
        "tie_rate_all": tie_rate(rows),
        "tie_rate_binding_necessary": tie_rate(bn_rows),
    }
    return out


def fmt(rep):
    bn, al = rep["binding_necessary"], rep["all"]
    return (f"BN(n={bn['n']}) group={bn['group']:.3f} match={bn['match']:.3f} text={bn['text']:.3f} "
            f"image={bn['image']:.3f} | ALL(n={al['n']}) group={al['group']:.3f} match={al['match']:.3f} | "
            f"pairAUC bn={rep['pair_auc_binding_necessary']:.3f} all={rep['pair_auc_all']:.3f} | "
            f"ties bn={rep['tie_rate_binding_necessary']:.2f}")


def proxy_feature_groups(groups, enc):
    """ORACLE CONTROL: metadata proxy tokens (v0) wrapped in the same container."""
    out = []
    for eg in encode_groups(groups, enc):
        t = lambda v: torch.tensor(v, dtype=DTYPE)  # noqa: E731
        img = torch.stack([t(eg.img[0]), t(eg.img[1])])
        out.append(FeatGroup(eg.gid, eg.kind_label, eg.same_words, img, img.mean(1), [t(x) for x in eg.txt],
                             [t(x) for x in eg.txt_hp], torch.stack([t(x).mean(0) for x in eg.txt]),
                             torch.stack([t(x).mean(0) for x in eg.txt_hp]), tuple(eg.group.captions),
                             eg.captions_hp, eg.group))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pixel_baseline_v1.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    dcfg = json.loads((ROOT / cfg["data_config"]).read_text())
    ecfg = json.loads((ROOT / cfg["encoder_config"]).read_text())
    pcfg = json.loads((ROOT / cfg["oracle_token_control_encoder_config"]).read_text())
    out = Path(args.out) if args.out else ROOT / "runs" / f"{cfg['name']}_{utc_stamp()}"
    caps = cfg["caps"]
    budget = CpuBudget(ROOT / caps["ledger"], caps["cpu_s_total_stage"])
    cap_info = budget.start(cfg["name"], max_as_bytes=int(caps["max_address_space_gib"] * 2**30))
    status = "FAILED"
    try:
        with RunRecorder(cfg["name"], out) as rec:
            rec.extra.update(config_sha256=sha256_json(cfg), data_config_sha256=sha256_json(dcfg),
                             encoder_config_sha256=sha256_json(ecfg), caps=cap_info,
                             torch_threads=torch.get_num_threads(), torch_version=torch.__version__)
            (out / "config.json").write_text(json.dumps({"stage": cfg, "data": dcfg, "encoder": ecfg}, indent=2))
            timing = {}
            metrics, decisions = {}, {}

            # ---- data panels ---------------------------------------------------
            ds, _ = build_dataset(dcfg)
            rec.extra["data_sha256"] = dataset_hash(ds)
            train_p = ds["train"][: cfg["panel"]["train"]["first_n_groups"]]
            dev_p = ds["dev"][: cfg["panel"]["dev"]["first_n_groups"]]
            fit_p = train_p[: cfg["panel"]["fit_subset"]["first_n_groups"]]
            panel = {k: {"gids": [g.gid for g in v], "kinds": {kk: sum(g.kind_label == kk for g in v)
                                                               for kk in sorted({g.kind_label for g in v})}}
                     for k, v in (("train", train_p), ("dev", dev_p), ("fit", fit_p))}
            rec.extra["panel"] = panel
            rec.log(f"data_v1 sha256 {rec.extra['data_sha256']} | panels {json.dumps({k: v['kinds'] for k, v in panel.items()})}")

            # ---- rendering + integrity -----------------------------------------
            t0 = time.process_time()
            imgs_train = render_members(train_p)
            imgs_dev = render_members(dev_p)
            imgs_dev_nuis = render_members(dev_p, seed_offset="nuisance2")
            bn_dev = [g for g in dev_p if binding_necessary(g.kind_label)]
            imgs_same_seed = [pixel_render.render(g.scenes[m], g.render_seeds[0]) for g in bn_dev for m in (0, 1)]
            mism = 0
            n_img = 0
            for groups, imgs in ((train_p, imgs_train), (dev_p, imgs_dev), (dev_p, imgs_dev_nuis)):
                for k, im in enumerate(imgs):
                    g, m = groups[k // 2], k % 2
                    n_img += 1
                    mism += pixel_render.decode(im) != pixel_render.scene_signature(g.scenes[m])
            for k, im in enumerate(imgs_same_seed):
                n_img += 1
                mism += pixel_render.decode(im) != pixel_render.scene_signature(bn_dev[k // 2].scenes[k % 2])
            timing["render_and_audit_cpu_s"] = round(time.process_time() - t0, 2)
            metrics["renderer_audit"] = {"images": n_img, "decode_mismatches": mism}
            rec.log(f"renderer audit: {n_img} images, {mism} decode mismatches")
            sheet = pixel_render.Image.new("RGB", (448, 224 * 4))
            for i in range(4):
                sheet.paste(imgs_dev[2 * i], (0, 224 * i))
                sheet.paste(imgs_dev[2 * i + 1], (224, 224 * i))
            sheet.save(out / "dev_examples.png")

            # ---- frozen encoder --------------------------------------------------
            t0 = time.process_time()
            enc = ClipEncoder(ecfg, ROOT)
            timing["encoder_load_cpu_s"] = round(time.process_time() - t0, 2)
            rec.extra["encoder"] = {"arch": ecfg["arch"], "checkpoint_sha256": ecfg["sha256"],
                                    "revision": ecfg["revision"],
                                    "n_params_frozen": sum(p.numel() for p in enc.model.parameters())}
            t0 = time.process_time()
            train_raw = build_feature_groups(train_p, enc, images=imgs_train, log=rec.log)
            dev_raw = build_feature_groups(dev_p, enc, images=imgs_dev, log=rec.log)
            nuis_pooled, _ = enc.encode_images(imgs_dev_nuis)
            same_pooled, same_tok = enc.encode_images(imgs_same_seed)
            timing["encoder_features_cpu_s"] = round(time.process_time() - t0, 2)
            n_enc_images = len(imgs_train) + len(imgs_dev) + len(imgs_dev_nuis) + len(imgs_same_seed)
            timing["encoder_images"] = n_enc_images
            timing["encoder_captions"] = 4 * (len(train_p) + len(dev_p))
            rec.log(f"encoder: {n_enc_images} images + {timing['encoder_captions']} captions, "
                    f"{timing['encoder_features_cpu_s']} CPU s (load {timing['encoder_load_cpu_s']} s)")
            allf = [fg.img for fg in train_raw + dev_raw] + [t for fg in train_raw + dev_raw for t in fg.txt]
            feat_ok = all(bool(torch.isfinite(x).all()) for x in allf)
            img_std_min = min(float(fg.img[m].std(0).mean()) for fg in train_raw + dev_raw for m in (0, 1))
            metrics["feature_integrity"] = {"all_finite": feat_ok, "min_mean_token_std_image": img_std_min}
            stats = standardizer(train_raw)
            rec.extra["standardizer_sha256"] = sha256_json({k: [v[0].tolist(), v[1].tolist()] for k, v in stats.items()})
            train_f = apply_standardizer(train_raw, stats)
            dev_f = apply_standardizer(dev_raw, stats)
            fit_f = train_f[: len(fit_p)]

            # ---- diagnostics ----------------------------------------------------
            nuis = {}
            for k, fg in enumerate(dev_raw):
                d_edit = cosine_distance(fg.img_pooled[0], fg.img_pooled[1])
                d_nuis = mean(cosine_distance(fg.img_pooled[m], nuis_pooled[2 * k + m]) for m in (0, 1))
                nuis.setdefault(fg.kind_label, {"edit": [], "nuisance": []})
                nuis[fg.kind_label]["edit"].append(d_edit)
                nuis[fg.kind_label]["nuisance"].append(d_nuis)
            metrics["nuisance_ratio"] = {k: {"median_edit": median(v["edit"]), "median_nuisance": median(v["nuisance"]),
                                             "ratio": median(v["edit"]) / median(v["nuisance"]), "n": len(v["edit"])}
                                         for k, v in sorted(nuis.items())}
            rec.log(f"nuisance ratio (edit/nuisance, pooled CLIP): "
                    f"{json.dumps({k: round(v['ratio'], 2) for k, v in metrics['nuisance_ratio'].items()})}")
            a, sc, lab = base_vs_edited_detectability(train_raw, dev_raw)
            metrics["base_vs_edited_detectability"] = {"auc": a, "auc_ci95": auc_bootstrap_ci(sc, lab, 500), "n_eval": len(lab)}
            rec.log(f"base-vs-edited pooled-image detectability AUC={a:.3f} CI95={metrics['base_vs_edited_detectability']['auc_ci95']}")

            # ---- reference controls --------------------------------------------
            for name, scorer in (("oracle", OracleScorer()), ("zero_shot_clip", ZeroShotClipScorer()),
                                 ("text_only", TextOnlyScorer().fit(train_f)),
                                 ("image_only_clip", ClipImageOnlyScorer().fit(train_raw))):
                metrics[name] = report(scorer, dev_f if name != "image_only_clip" else dev_raw)
                rec.log(f"dev {name}: {fmt(metrics[name])}")
            mc = [summarize(evaluate_groups(RandomScorer(seed=s), dev_f))["all"] for s in range(500)]
            metrics["random_mc500"] = {m: mean(r[m] for r in mc) for m in ("text", "image", "group", "match", "aug")}
            metrics["chance_analytic"] = CHANCE

            # ---- fit sanity -----------------------------------------------------
            tcfg = dict(cfg["train"])
            hc = cfg["head"]
            fs = cfg["fit_sanity"]
            head = TorchFactorizedHead(d=512, hc=hc["hc"], hb=hc["hb"], window=hc["window"], seed=fs["seed"],
                                       init_scale=hc["init_scale"])
            rec.extra["head_n_params"] = head.n_params()
            rec.log(f"fit sanity: {len(fit_f)} groups, {fs['steps']} steps (head params {head.n_params()})")
            hist = train(head, fit_f, dict(tcfg, steps=fs["steps"]), hard_neg=True, seed=fs["seed"], log=rec.log)
            metrics["fit_sanity"] = {"train_fit": report(TorchHeadScorer(head), fit_f), "loss_first": hist[0]["task"],
                                     "loss_last": hist[-1]["task"]}
            rec.log(f"fit sanity train-fit: {fmt(metrics['fit_sanity']['train_fit'])}")

            # ---- hard-negative baseline, fixed setting, fixed seeds -------------
            runs = {}
            for seed in cfg["seeds"]:
                head = TorchFactorizedHead(d=512, hc=hc["hc"], hb=hc["hb"], window=hc["window"], seed=seed,
                                           init_scale=hc["init_scale"])
                import random as _r
                order = list(range(len(train_f)))
                _r.Random(f"train-order:{seed}").shuffle(order)
                first_batch = [train_f[i] for i in reversed(order[-tcfg["batch_groups"]:])]

                def ratio():
                    _, lt, le = batch_loss(head, first_batch, True, 1.0, tcfg["logit_scale"], 0.1)
                    return grad_norm_of(head, le) / grad_norm_of(head, lt)

                r_init = ratio()
                t0 = time.process_time()
                rec.log(f"hardneg seed {seed}: training {tcfg['steps']} steps on {len(train_f)} groups")
                hist = train(head, train_f, tcfg, hard_neg=True, seed=seed, log=rec.log)
                t_train = time.process_time() - t0
                r_end = ratio()
                res = {"param_sha256": sha256_json({k: v.tolist() for k, v in head.p.items()}),
                       "train_cpu_s": round(t_train, 2), "loss_first": hist[0]["task"], "loss_last": hist[-1]["task"],
                       "grad_ratio_edit_over_task": {"init": r_init, "end": r_end},
                       "train_fit": report(TorchHeadScorer(head), train_f),
                       "history": hist}
                for ch in ("all", "content", "binding"):
                    res[f"dev[{ch}]"] = report(TorchHeadScorer(head, ch), dev_f)
                res["dev[image_shuffled]"] = report(TorchHeadScorer(head), shuffled(dev_f, "image", seed))
                res["dev[text_shuffled]"] = report(TorchHeadScorer(head), shuffled(dev_f, "text", seed))
                with torch.no_grad():
                    std_same = (same_tok.to(DTYPE) - stats["img"][0]) / stats["img"][1]
                    c, _ = head.encode_image(std_same)
                    resid = [float((c[2 * k + 1] - c[2 * k]).norm() / c[2 * k].norm()) for k in range(len(bn_dev))]
                res["content_residual_binding_necessary"] = {"median": median(resid), "max": max(resid), "n": len(resid)}
                torch.save({k: v.detach() for k, v in head.p.items()}, out / f"head_hardneg_seed{seed}.pt")
                runs[f"seed{seed}"] = res
                rec.log(f"  seed {seed}: loss {res['loss_first']:.3f}->{res['loss_last']:.3f} "
                        f"grad ratio init={r_init:.2f} end={r_end:.2f} content residual median={median(resid):.3f}")
                rec.log(f"  seed {seed} train-fit: {fmt(res['train_fit'])}")
                for key in ("dev[all]", "dev[content]", "dev[binding]", "dev[image_shuffled]", "dev[text_shuffled]"):
                    rec.log(f"  seed {seed} {key}: {fmt(res[key])}")
            metrics["hardneg"] = runs

            # ---- oracle-object-token control ------------------------------------
            penc = ProxyEncoder(**{k: pcfg[k] for k in ("dim", "seed", "jitter", "lighting", "pos_scale")})
            ptrain_raw = proxy_feature_groups(train_p, penc)
            pdev_raw = proxy_feature_groups(dev_p, penc)
            pstats = standardizer(ptrain_raw)
            ptrain, pdev = apply_standardizer(ptrain_raw, pstats), apply_standardizer(pdev_raw, pstats)
            oruns = {}
            for seed in cfg["seeds"]:
                head = TorchFactorizedHead(d=pcfg["dim"], hc=hc["hc"], hb=hc["hb"], window=hc["window"], seed=seed,
                                           init_scale=hc["init_scale"])
                hist = train(head, ptrain, tcfg, hard_neg=True, seed=seed)
                oruns[f"seed{seed}"] = {"loss_first": hist[0]["task"], "loss_last": hist[-1]["task"],
                                        "train_fit": report(TorchHeadScorer(head), ptrain),
                                        "dev": report(TorchHeadScorer(head), pdev)}
                oruns[f"seed{seed}"]["dev"]["provenance"] = "ORACLE metadata proxy tokens + proxy text (control)"
                rec.log(f"  oracle-token control seed {seed} dev: {fmt(oruns[f'seed{seed}']['dev'])}")
            metrics["oracle_token_control"] = oruns

            # ---- decisions (rules fixed in the config) ---------------------------
            dev_cells = [metrics[k] for k in ("oracle", "zero_shot_clip", "text_only", "image_only_clip")]
            dev_cells += [runs[s][k] for s in runs for k in runs[s] if k.startswith("dev[")]
            dev_cells += [oruns[s]["dev"] for s in oruns]
            complete = all(c["binding_necessary"] and c["all"]["n"] == len(dev_p) for c in dev_cells)
            oracle_ok = all(metrics["oracle"][part][m] == 1.0 for part in ("binding_necessary", "all")
                            for m in ("text", "image", "group", "match", "aug"))
            decisions["Q1_pixel_input_and_eval_path"] = {
                "renderer_mismatches": mism, "features_finite": feat_ok, "min_image_token_std": img_std_min,
                "oracle_control_all_1.0": oracle_ok, "all_cells_present": complete,
                "provenance_guard_test": "see final unittest run (tests/test_pixel_path.py)",
                "verdict": "PASS_PENDING_FINAL_TESTS" if (mism == 0 and feat_ok and img_std_min > 0 and oracle_ok and complete) else "FAIL"}
            fit_group = metrics["fit_sanity"]["train_fit"]["all"]["group"]
            loss_ratio = mean(runs[s]["loss_last"] / runs[s]["loss_first"] for s in runs)
            decisions["Q2_baseline_trains"] = {
                "fit_sanity_group": fit_group, "fit_rule": fs["pass_rule"], "mean_loss_ratio": loss_ratio,
                "trains_rule": cfg["trains_rule"],
                "verdict": "PASS" if (fit_group >= 0.75 and loss_ratio <= 0.7) else "FAIL"}
            n_bn = metrics["oracle"]["binding_necessary"]["n"]
            decisions["Q3_binding_necessary_evaluable"] = {
                "n_binding_necessary_dev": n_bn, "oracle_on_subset": metrics["oracle"]["binding_necessary"]["group"],
                "verdict": "PASS" if (n_bn >= 12 and metrics["oracle"]["binding_necessary"]["group"] == 1.0 and complete) else "FAIL"}
            rec.log(f"decisions: {json.dumps({k: v['verdict'] for k, v in decisions.items()})}")
            (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
            (out / "decisions.json").write_text(json.dumps(decisions, indent=2))
            rec.extra["timing"] = timing
            rec.extra["decisions"] = {k: v["verdict"] for k, v in decisions.items()}
            rec.extra["software_status"] = "COMPLETED"
            rec.extra["science_status"] = "SCIENCE_NOT_EVALUATED"
        status = "COMPLETED"
    except BudgetExceeded as e:
        status = f"CAP_EXCEEDED: {e}"
        raise
    except MemoryError as e:
        status = f"OOM: {e}"
        raise
    finally:
        entry = budget.finish(status, out)
        print(json.dumps(entry), f"| ledger used {budget.used():.1f}/{budget.cap_s:.0f}s")


if __name__ == "__main__":
    main()
