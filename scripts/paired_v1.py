#!/usr/bin/env python3
"""Stage paired_v1: hard-negative (A) vs hard-negative + edit-consistency (B), paired.

Everything is fixed by configs/paired_v1.json before this script runs: panel,
lambda, update budget, seeds, evaluation, statistics and verdict rules. Both arms
share the same cached frozen-encoder features, initialization, batches, masks and
optimizer. The development panel is scored only at the final step. Runs under the
task CPU ledger with one torch thread and a 3 GiB address-space cap.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import resource  # noqa: E402
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
from bindcomp.evaluate import auc, binding_necessary, evaluate_groups, pair_scores, summarize, tie_rate  # noqa: E402
from bindcomp.manifest import RunRecorder, sha256_json, utc_stamp  # noqa: E402
from bindcomp.pixel import (ZeroShotClipScorer, apply_standardizer, build_feature_groups,  # noqa: E402
                            render_members, shuffled, standardizer)
from bindcomp.shortcuts import OracleScorer, RandomScorer  # noqa: E402
from bindcomp.torch_head import (DTYPE, TorchFactorizedHead, TorchHeadScorer, batch_embeddings,  # noqa: E402
                                 edit_consistency, grad_norm_of, info_nce, negative_mask, train)

METRICS = ("text", "image", "group", "match", "aug", "aug_group")


def subset_rows(rows, fgs, pred):
    keep = [i for i, fg in enumerate(fgs) if pred(fg.kind_label)]
    return [rows[i] for i in keep], [fgs[i] for i in keep]


def summary(scorer, fgs):
    rows = evaluate_groups(scorer, fgs)
    out = {"rows": rows}
    for name, pred in (("bn", binding_necessary), ("binding", lambda k: k == "binding"),
                       ("relation", lambda k: k == "relation"), ("all", lambda k: True)):
        r, f = subset_rows(rows, fgs, pred)
        s = summarize(r).get("all", {})
        sc, lab = pair_scores(r, f)
        out[name] = {**{m: s.get(m) for m in METRICS}, "n": len(r), "pair_auc": auc(sc, lab) if r else None,
                     "tie_rate": tie_rate(r)}
    return out


def strip_rows(s):
    return {k: v for k, v in s.items() if k != "rows"}


def boot_mean(values, n_boot, rng):
    n = len(values)
    return sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))


def ci(sorted_vals):
    n = len(sorted_vals)
    return [sorted_vals[int(0.025 * n)], sorted_vals[min(n - 1, int(0.975 * n))]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paired_v1.json")
    args = ap.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    dcfg = json.loads((ROOT / cfg["data_config"]).read_text())
    ecfg = json.loads((ROOT / cfg["encoder_config"]).read_text())
    out = ROOT / "runs" / f"{cfg['name']}_{utc_stamp()}"
    caps = cfg["caps"]
    budget = CpuBudget(ROOT / caps["ledger"], caps["cpu_s_total_task"])
    cap_info = budget.start(cfg["name"], max_as_bytes=int(caps["max_address_space_gib"] * 2**30))
    status = "FAILED"
    try:
        with RunRecorder(cfg["name"], out) as rec:
            rec.extra.update(config_sha256=sha256_json(cfg), data_config_sha256=sha256_json(dcfg),
                             encoder_config_sha256=sha256_json(ecfg), caps=cap_info,
                             torch_threads=torch.get_num_threads(), torch_version=torch.__version__)
            (out / "config.json").write_text(json.dumps({"paired": cfg, "data": dcfg, "encoder": ecfg}, indent=2))
            timing = {}

            # ---- panel -------------------------------------------------------------
            ds, _ = build_dataset(dcfg)
            rec.extra["data_sha256"] = dataset_hash(ds)
            P = cfg["panel"]
            train_p = ds["train"][P["train"]["start"]:P["train"]["start"] + P["train"]["n_groups"]]
            dev_p = ds["dev"][P["dev"]["start"]:P["dev"]["start"] + P["dev"]["n_groups"]]
            gids_sha = sha256_json({"train": [g.gid for g in train_p], "dev": [g.gid for g in dev_p]})
            rec.extra["panel_gids_sha256"] = gids_sha
            rec.log(f"data_v1 {rec.extra['data_sha256'][:16]} | panel train {len(train_p)} dev {len(dev_p)} "
                    f"(dev {dev_p[0].gid}..{dev_p[-1].gid}) gids sha {gids_sha[:16]}")

            # ---- render + audit ----------------------------------------------------
            t0 = time.process_time()
            imgs_train, imgs_dev = render_members(train_p), render_members(dev_p)
            bn_dev = [g for g in dev_p if binding_necessary(g.kind_label)]
            imgs_same = [pixel_render.render(g.scenes[m], g.render_seeds[0]) for g in bn_dev for m in (0, 1)]
            mism = 0
            for groups, imgs in ((train_p, imgs_train), (dev_p, imgs_dev)):
                for k, im in enumerate(imgs):
                    mism += pixel_render.decode(im) != pixel_render.scene_signature(groups[k // 2].scenes[k % 2])
            for k, im in enumerate(imgs_same):
                mism += pixel_render.decode(im) != pixel_render.scene_signature(bn_dev[k // 2].scenes[k % 2])
            n_img = len(imgs_train) + len(imgs_dev) + len(imgs_same)
            timing["render_and_audit_cpu_s"] = round(time.process_time() - t0, 2)
            rec.log(f"renderer audit: {n_img} images, {mism} decode mismatches ({timing['render_and_audit_cpu_s']} CPU s)")

            # ---- frozen encoder cache (shared by both arms) ------------------------
            t0 = time.process_time()
            enc = ClipEncoder(ecfg, ROOT)
            train_raw = build_feature_groups(train_p, enc, images=imgs_train, log=rec.log, text_batch=64)
            dev_raw = build_feature_groups(dev_p, enc, images=imgs_dev, log=rec.log, text_batch=64)
            _, same_tok = enc.encode_images(imgs_same)
            del imgs_train, imgs_dev, imgs_same
            timing["encoder_cpu_s"] = round(time.process_time() - t0, 2)
            timing["encoder_images"] = n_img
            timing["encoder_captions"] = 4 * (len(train_p) + len(dev_p))
            rec.log(f"encoder: {n_img} images + {timing['encoder_captions']} captions, {timing['encoder_cpu_s']} CPU s")
            del enc
            stats = standardizer(train_raw)
            rec.extra["standardizer_sha256"] = sha256_json({k: [v[0].tolist(), v[1].tolist()] for k, v in stats.items()})
            train_f, dev_f = apply_standardizer(train_raw, stats), apply_standardizer(dev_raw, stats)
            del train_raw, dev_raw  # pooled embeddings are carried over unchanged into *_f
            same_std = (same_tok.to(DTYPE) - stats["img"][0]) / stats["img"][1]

            # ---- controls ------------------------------------------------------------
            controls = {"oracle": strip_rows(summary(OracleScorer(), dev_f)),
                        "zero_shot_clip": strip_rows(summary(ZeroShotClipScorer(), dev_f))}
            mc = [summary(RandomScorer(seed=s), dev_f) for s in range(300)]
            controls["random_mc300_bn"] = {m: mean(r["bn"][m] for r in mc) for m in ("text", "image", "group", "match", "aug", "aug_group")}
            controls["random_mc300_bn"]["pair_auc"] = mean(r["bn"]["pair_auc"] for r in mc)
            rec.log(f"controls: oracle bn group {controls['oracle']['bn']['group']}, zero-shot bn group "
                    f"{controls['zero_shot_clip']['bn']['group']:.3f} match {controls['zero_shot_clip']['bn']['match']:.3f}; "
                    f"random bn group {controls['random_mc300_bn']['group']:.3f}")

            # ---- paired training -------------------------------------------------------
            tcfg, hc = cfg["train"], cfg["head"]
            arms = {"A": {}, "B": {}}
            rows_store = {"A": {}, "B": {}}
            for seed in cfg["seeds"]:
                order = list(range(len(train_f)))
                random.Random(f"train-order:{seed}").shuffle(order)
                first_batch = [train_f[i] for i in reversed(order[-tcfg["batch_groups"]:])]
                for arm in ("A", "B"):
                    lam = cfg["arms"][arm]["lam_eq"]
                    head = TorchFactorizedHead(d=hc["d"], hc=hc["hc"], hb=hc["hb"], window=hc["window"], seed=seed,
                                               init_scale=hc["init_scale"])
                    init_hash = sha256_json({k: v.tolist() for k, v in head.p.items()})
                    eI, eT = batch_embeddings(head, first_batch)
                    lt = info_nce(eI @ eT.T, negative_mask(first_batch, True), tcfg["logit_scale"])
                    le = edit_consistency(eI[1::2] - eI[0::2], eT[1::2] - eT[0::2], cfg["arms"]["B"]["tau_eq"])
                    g_task, g_edit = grad_norm_of(head, lt), grad_norm_of(head, le)
                    init = {"loss_task": lt.item(), "loss_edit": le.item(), "grad_task": g_task, "grad_edit": g_edit,
                            "grad_ratio": g_edit / g_task, "lam_scaled_grad_ratio": lam * g_edit / g_task,
                            "lam_scaled_loss_ratio": lam * le.item() / lt.item(), "init_param_sha256": init_hash}
                    rec.log(f"[{arm} seed {seed}] init: L_task={lt.item():.3f} L_edit={le.item():.3f} "
                            f"grad ratio edit/task={g_edit / g_task:.2f} (x lam={lam}: {lam * g_edit / g_task:.3f})")
                    t0 = time.process_time()
                    hist = train(head, train_f, tcfg, True, seed, log=rec.log, lam_eq=lam,
                                 tau_eq=cfg["arms"]["B"]["tau_eq"], monitor_edit=True,
                                 grad_split_every=tcfg["grad_split_every"])
                    t_train = time.process_time() - t0
                    dev_all = summary(TorchHeadScorer(head), dev_f)
                    res = {"init": init, "train_cpu_s": round(t_train, 2),
                           "param_sha256": sha256_json({k: v.tolist() for k, v in head.p.items()}),
                           "loss_first": hist[0]["task"], "loss_last": hist[-1]["task"],
                           "edit_first": hist[0]["edit"], "edit_last": hist[-1]["edit"],
                           "grad_split": [{k: h[k] for k in ("step", "grad_task", "grad_edit")} for h in hist if "grad_edit" in h],
                           "history": [{k: h[k] for k in ("step", "task", "edit", "loss", "grad_norm")} for h in hist
                                       if h["step"] == 1 or h["step"] % tcfg["log_every"] == 0],
                           "dev": strip_rows(dev_all),
                           "dev_content": strip_rows(summary(TorchHeadScorer(head, "content"), dev_f)),
                           "dev_binding": strip_rows(summary(TorchHeadScorer(head, "binding"), dev_f)),
                           "dev_image_shuffled": strip_rows(summary(TorchHeadScorer(head), shuffled(dev_f, "image", seed))),
                           "dev_text_shuffled": strip_rows(summary(TorchHeadScorer(head), shuffled(dev_f, "text", seed))),
                           "train_fit": strip_rows(summary(TorchHeadScorer(head), train_f))}
                    with torch.no_grad():
                        c, _ = head.encode_image(same_std)
                        resid = [float((c[2 * k + 1] - c[2 * k]).norm() / c[2 * k].norm()) for k in range(len(bn_dev))]
                    res["content_residual_bn"] = {"median": median(resid), "max": max(resid), "n": len(resid)}
                    torch.save({k: v.detach() for k, v in head.p.items()}, out / f"head_{arm}_seed{seed}.pt")
                    arms[arm][f"seed{seed}"] = res
                    rows_store[arm][f"seed{seed}"] = dev_all["rows"]
                    d = res["dev"]
                    rec.log(f"[{arm} seed {seed}] loss {res['loss_first']:.3f}->{res['loss_last']:.3f} edit "
                            f"{res['edit_first']:.3f}->{res['edit_last']:.3f} | dev BN group {d['bn']['group']:.3f} "
                            f"match {d['bn']['match']:.3f} auc {d['bn']['pair_auc']:.3f} | all group {d['all']['group']:.3f} "
                            f"| train BN group {res['train_fit']['bn']['group']:.3f} | {t_train:.1f} CPU s")

            # ---- paired statistics ----------------------------------------------------------
            seeds = [f"seed{s}" for s in cfg["seeds"]]
            bn_idx = [i for i, fg in enumerate(dev_f) if binding_necessary(fg.kind_label)]
            rng = random.Random("paired-bootstrap:0")
            n_boot = 2000
            paired = {}
            arm_ci = {"A": {}, "B": {}}
            for metric in ("group", "match"):
                per = {a: [mean(rows_store[a][s][i][metric] for s in seeds) for i in bn_idx] for a in ("A", "B")}
                diffs = [b - a for a, b in zip(per["A"], per["B"])]
                per_seed = [arms["B"][s]["dev"]["bn"][metric] - arms["A"][s]["dev"]["bn"][metric] for s in seeds]
                paired[f"dev_bn_{metric}"] = {"mean_diff": mean(diffs), "ci95": ci(boot_mean(diffs, n_boot, rng)),
                                              "per_seed_diff": per_seed, "n_seeds_positive": sum(d > 0 for d in per_seed),
                                              "n_groups": len(bn_idx)}
                for a in ("A", "B"):
                    arm_ci[a][metric] = {"mean": mean(per[a]), "ci95": ci(boot_mean(per[a], n_boot, rng))}
            # pair AUC: resample groups, recompute AUC per arm and seed (oracle labels precomputed once)
            bn_fgs = [dev_f[i] for i in bn_idx]
            pre = {}
            for a in ("A", "B"):
                for s in seeds:
                    per_group = []
                    for j, i in enumerate(bn_idx):
                        sc, lab = pair_scores([rows_store[a][s][i]], [bn_fgs[j]])
                        per_group.append((sc, lab))
                    pre[(a, s)] = per_group
            diffs_auc = []
            for _ in range(n_boot):
                pick = [rng.randrange(len(bn_idx)) for _ in bn_idx]
                vals = []
                for s in seeds:
                    per_arm = []
                    for a in ("A", "B"):
                        sc = [x for j in pick for x in pre[(a, s)][j][0]]
                        lab = [y for j in pick for y in pre[(a, s)][j][1]]
                        per_arm.append(auc(sc, lab))
                    vals.append(per_arm[1] - per_arm[0])
                diffs_auc.append(mean(vals))
            diffs_auc.sort()
            per_seed_auc = [arms["B"][s]["dev"]["bn"]["pair_auc"] - arms["A"][s]["dev"]["bn"]["pair_auc"] for s in seeds]
            paired["dev_bn_auc"] = {"mean_diff": mean(per_seed_auc), "ci95": ci(diffs_auc), "per_seed_diff": per_seed_auc,
                                    "n_seeds_positive": sum(d > 0 for d in per_seed_auc), "n_groups": len(bn_idx)}

            # ---- verdict (rules fixed in the config) -------------------------------------------
            fits = {a: mean(arms[a][s]["train_fit"]["bn"]["group"] for s in seeds) >= 0.75 for a in ("A", "B")}
            at_chance = {a: arm_ci[a]["group"]["ci95"][0] <= 1 / 6 and arm_ci[a]["match"]["ci95"][0] <= 0.5
                         for a in ("A", "B")}
            g = paired["dev_bn_group"]
            if not (fits["A"] or fits["B"]):
                label = "OPTIMIZATION_OR_INPUT_UNRESOLVED"
            elif at_chance["A"] and at_chance["B"]:
                label = "GENERALIZATION_UNRESOLVED"
            elif g["mean_diff"] >= 0.05 and g["ci95"][0] > 0 and g["n_seeds_positive"] == len(seeds):
                label = "PRELIMINARY_ADDED_UTILITY"
            else:
                label = "EDIT_LOSS_BRANCH_ON_HOLD"
            branch = "CONTINUE_WITH_CAUTION" if label == "PRELIMINARY_ADDED_UTILITY" else "ON_HOLD"
            verdict = {"label": label, "edit_loss_branch": branch, "fits": fits, "at_chance": at_chance,
                       "arm_bn_ci": arm_ci}
            rec.log(f"verdict: {json.dumps({'label': label, 'fits': fits, 'at_chance': at_chance})}")
            rec.log(f"paired BN group diff {g['mean_diff']:.3f} CI {g['ci95']} per-seed {g['per_seed_diff']}")

            def flat(a, s):
                r = arms[a][s]
                return {"dev_bn_group": r["dev"]["bn"]["group"], "dev_bn_match": r["dev"]["bn"]["match"],
                        "dev_bn_auc": r["dev"]["bn"]["pair_auc"], "dev_all_group": r["dev"]["all"]["group"],
                        "dev_all_match": r["dev"]["all"]["match"], "train_bn_group": r["train_fit"]["bn"]["group"],
                        "dev_binding_group": r["dev"]["binding"]["group"], "dev_relation_group": r["dev"]["relation"]["group"],
                        "dev_content_all_group": r["dev_content"]["all"]["group"], "loss_last": r["loss_last"],
                        "edit_last": r["edit_last"], "content_residual_bn": r["content_residual_bn"]["median"]}

            ru = resource.getrusage(resource.RUSAGE_SELF)
            summary_out = {
                "panel": {"train_groups": len(train_p), "dev_groups": len(dev_p), "dev_binding_necessary": len(bn_idx),
                          "gids_sha256": gids_sha},
                "config": {"lam_eq": cfg["arms"]["B"]["lam_eq"], "steps": tcfg["steps"], "seeds": cfg["seeds"]},
                "renderer_audit": {"images": n_img, "decode_mismatches": mism},
                "controls": controls,
                "arms": {a: {s: flat(a, s) for s in seeds} for a in ("A", "B")},
                "paired": paired, "verdict": verdict,
                "grad_ratio": {"init_mean": mean(arms["A"][s]["init"]["grad_ratio"] for s in seeds),
                               "init_lam_scaled_mean": mean(arms["B"][s]["init"]["lam_scaled_grad_ratio"] for s in seeds)},
                "cost": {"stage_cpu_s": round(ru.ru_utime + ru.ru_stime, 1), "peak_rss_mib": round(ru.ru_maxrss / 1024, 1),
                         **timing},
            }
            (out / "paired_summary.json").write_text(json.dumps(summary_out, indent=1, default=str))
            (out / "arms_detail.json").write_text(json.dumps(arms, indent=1, default=str))
            rec.extra.update(timing=timing, verdict=label, software_status="COMPLETED",
                             science_status="DEV_ONLY_PAIRED_OBSERVATION")
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
