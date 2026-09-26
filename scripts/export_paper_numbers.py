#!/usr/bin/env python3
"""Export every number used in the manuscript from raw run files.

Writes
  paper/generated/numbers.tex      LaTeX macros (the ONLY source of numbers in the text)
  paper/generated/provenance.tsv   macro -> file -> JSON field (or derivation rule) -> value
  paper/tables/*.tex               tables built from the same values
Sources are fixed run directories; optional ones (paired comparison, split analysis)
are used only if present. Nothing is typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SMALL = "runs/pixel_baseline_v1_20260926T161957Z"
META_V1 = "runs/metadata_check_data_v1_20260926T161932Z"
META_V0 = "runs/metadata_check_data_v0_20260926T152348Z"
SMOKE = "runs/smoke_v0_20260926T152653Z"
SEEDS = ("seed0", "seed1", "seed2")
SEED_WORDS = {"seed0": "Zero", "seed1": "One", "seed2": "Two"}

macros, prov = [], []


def load(rel):
    return json.loads((ROOT / rel).read_text())


def get(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def fmt(v, kind):
    if v is None:
        return "n/a"
    if kind == "int":
        return f"{int(v):,}".replace(",", "{,}")
    if kind == "s3":
        return f"{v:.3f}"
    if kind == "c3":  # compact table cell: .062 (no leading zero), 1.00 for one
        return "1.00" if abs(v - 1.0) < 5e-4 else f"{v:.3f}".lstrip("0")
    if kind == "s2":
        return f"{v:.2f}"
    if kind == "s1":
        return f"{v:.1f}"
    if kind == "pct":
        return f"{100 * v:.1f}\\%"
    if kind == "e2":
        from math import log10
        k = round(log10(v)) if v > 0 else None
        if k is not None and abs(v - 10 ** k) < 1e-12 * max(1.0, v):
            return f"$10^{{{k}}}$"
        return f"{v:.1e}"
    raise ValueError(kind)


def add(name, rel, path, kind="s3"):
    v = get(load(rel), path)
    macros.append((name, fmt(v, kind)))
    prov.append((name, rel, "/".join(map(str, path)), repr(v), "raw field"))
    return v


def add_derived(name, value, sources, rule, kind="s3"):
    macros.append((name, fmt(value, kind)))
    prov.append((name, sources, "-", repr(value), rule))
    return value


def seeds_range(prefix, rel, path_fn, kind="s3", root=None):
    """Per-seed values plus min/max macros; path_fn(seed) -> JSON path."""
    data = load(rel)
    vals = []
    for s in SEEDS:
        p = path_fn(s)
        v = get(data, p)
        vals.append(v)
        macros.append((f"{prefix}Seed{SEED_WORDS[s]}", fmt(v, kind)))
        prov.append((f"{prefix}Seed{SEED_WORDS[s]}", rel, "/".join(map(str, p)), repr(v), "raw field"))
    add_derived(f"{prefix}Min", min(vals), rel, f"min over seeds of {'/'.join(map(str, path_fn('<seed>')))}", kind)
    add_derived(f"{prefix}Max", max(vals), rel, f"max over seeds of {'/'.join(map(str, path_fn('<seed>')))}", kind)
    return vals


# ---------------------------------------------------------------------------
# Small development panel (pixel_baseline_v1, reference run)
# ---------------------------------------------------------------------------

def small_panel():
    m = f"{SMALL}/metrics.json"
    man = f"{SMALL}/manifest.json"
    tr = load(man)["panel"]
    add_derived("SPtrainGroups", len(tr["train"]["gids"]), man, "len(panel/train/gids)", "int")
    add_derived("SPdevGroups", len(tr["dev"]["gids"]), man, "len(panel/dev/gids)", "int")
    add("SPdevBNn", m, ["oracle", "binding_necessary", "n"], "int")
    add("SPrenderImages", m, ["renderer_audit", "images"], "int")
    add("SPrenderMismatch", m, ["renderer_audit", "decode_mismatches"], "int")
    add("SPchanceGroupMC", m, ["random_mc500", "group"])
    add("SPchanceMatchMC", m, ["random_mc500", "match"])
    add("SPoracleBNGroup", m, ["oracle", "binding_necessary", "group"])
    for pre, key in (("SPzs", "zero_shot_clip"), ("SPtxt", "text_only"), ("SPimg", "image_only_clip")):
        add(f"{pre}BNGroup", m, [key, "binding_necessary", "group"])
        add(f"{pre}BNMatch", m, [key, "binding_necessary", "match"])
        add(f"{pre}BNAUC", m, [key, "pair_auc_binding_necessary"])
        add(f"{pre}AllGroup", m, [key, "all", "group"])
        add(f"{pre}AllMatch", m, [key, "all", "match"])
    add("SPfitGroup", m, ["fit_sanity", "train_fit", "all", "group"])
    add("SPfitAugGroup", m, ["fit_sanity", "train_fit", "all", "aug_group"])
    add("SPfitN", m, ["fit_sanity", "train_fit", "all", "n"], "int")
    H = lambda k, *p: (lambda s: ["hardneg", s, k, *p])  # noqa: E731
    seeds_range("SPdevBNGroup", m, H("dev[all]", "binding_necessary", "group"))
    seeds_range("SPdevBNMatch", m, H("dev[all]", "binding_necessary", "match"))
    seeds_range("SPdevBNAUC", m, H("dev[all]", "pair_auc_binding_necessary"))
    seeds_range("SPdevAllGroup", m, H("dev[all]", "all", "group"))
    seeds_range("SPdevAllMatch", m, H("dev[all]", "all", "match"))
    seeds_range("SPcontentAllGroup", m, H("dev[content]", "all", "group"))
    seeds_range("SPcontentBNGroup", m, H("dev[content]", "binding_necessary", "group"))
    seeds_range("SPbindingBNGroup", m, H("dev[binding]", "binding_necessary", "group"))
    seeds_range("SPimgShufBNGroup", m, H("dev[image_shuffled]", "binding_necessary", "group"))
    seeds_range("SPtxtShufBNGroup", m, H("dev[text_shuffled]", "binding_necessary", "group"))
    seeds_range("SPtrainBNGroup", m, H("train_fit", "binding_necessary", "group"))
    seeds_range("SPtrainAllGroup", m, H("train_fit", "all", "group"))
    seeds_range("SPlossFirst", m, lambda s: ["hardneg", s, "loss_first"], "s2")
    seeds_range("SPlossLast", m, lambda s: ["hardneg", s, "loss_last"], "s2")
    seeds_range("SPgradInit", m, lambda s: ["hardneg", s, "grad_ratio_edit_over_task", "init"], "s1")
    seeds_range("SPgradEnd", m, lambda s: ["hardneg", s, "grad_ratio_edit_over_task", "end"], "s2")
    seeds_range("SPcontentResid", m, lambda s: ["hardneg", s, "content_residual_binding_necessary", "median"], "s2")
    O = lambda *p: (lambda s: ["oracle_token_control", s, *p])  # noqa: E731
    seeds_range("SPotcBNGroup", m, O("dev", "binding_necessary", "group"))
    seeds_range("SPotcBNMatch", m, O("dev", "binding_necessary", "match"))
    seeds_range("SPotcTrainBNGroup", m, O("train_fit", "binding_necessary", "group"))
    for kind in ("binding", "relation", "attribute", "object"):
        add(f"SPnuis{kind.capitalize()}", m, ["nuisance_ratio", kind, "ratio"], "s2")
    cfg = f"{SMALL}/config.json"
    add("SPlr", cfg, ["stage", "train", "lr"], "s2")
    add("SPwd", cfg, ["stage", "train", "weight_decay"], "e2")
    add("SPsteps", cfg, ["stage", "train", "steps"], "int")
    add("SPbatch", cfg, ["stage", "train", "batch_groups"], "int")
    add("SPdetectN", m, ["base_vs_edited_detectability", "n_eval"], "int")
    add("SPdetectAUC", m, ["base_vs_edited_detectability", "auc"], "s2")
    add("SPdetectLo", m, ["base_vs_edited_detectability", "auc_ci95", 0], "s2")
    add("SPdetectHi", m, ["base_vs_edited_detectability", "auc_ci95", 1], "s2")
    add("SPheadParams", man, ["head_n_params"], "int")
    add("SPencParams", man, ["encoder", "n_params_frozen"], "int")
    add("SPencCPU", man, ["timing", "encoder_features_cpu_s"], "s1")
    add("SPencImages", man, ["timing", "encoder_images"], "int")
    add("SPencCaptions", man, ["timing", "encoder_captions"], "int")
    add("SPstageCPU", man, ["cpu_time_s"], "s1")
    add("SPpeakRSS", man, ["peak_rss_mib"], "s1")

    data = load(m)
    getters = [("bn_group", lambda r: r["binding_necessary"]["group"]),
               ("bn_match", lambda r: r["binding_necessary"]["match"]),
               ("bn_auc", lambda r: r["pair_auc_binding_necessary"]),
               ("all_group", lambda r: r["all"]["group"]),
               ("all_match", lambda r: r["all"]["match"]),
               ("all_auc", lambda r: r["pair_auc_all"])]
    rows = [("Oracle (metadata)", [data["oracle"]]), ("Zero-shot CLIP", [data["zero_shot_clip"]]),
            ("Text-only (n-gram LR)", [data["text_only"]]), ("Image-only (CLIP LR)", [data["image_only_clip"]]),
            None]
    for label, k in (("Pixel head (hard-neg.)", "dev[all]"), ("\\quad content channel", "dev[content]"),
                     ("\\quad binding channel", "dev[binding]"), ("\\quad images shuffled", "dev[image_shuffled]"),
                     ("\\quad captions shuffled", "dev[text_shuffled]"), ("\\quad train-panel fit", "train_fit")):
        rows.append((label, [data["hardneg"][s][k] for s in SEEDS]))
    rows.append(None)
    rows.append(("Oracle object tokens", [data["oracle_token_control"][s]["dev"] for s in SEEDS]))
    for name, cols in (("small_panel.tex", ("bn_group", "bn_match", "bn_auc", "all_group")),
                       ("small_panel_all.tex", ("all_group", "all_match", "all_auc"))):
        lines = []
        for row in rows:
            if row is None:
                lines.append("\\midrule")
                continue
            label, recs = row
            cells = ["/".join(fmt(g(r), "c3") for r in recs) for key, g in getters if key in cols]
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        write_table(name, lines, SMALL)


# ---------------------------------------------------------------------------
# Data audits
# ---------------------------------------------------------------------------

def data_audits():
    add("DVauditFail", f"{META_V1}/audit.json", ["n_fail"], "int")
    add("DVblindText", f"{META_V1}/blind.json", ["text", "auc"], "s3")
    add("DVblindImage", f"{META_V1}/blind.json", ["image", "auc"], "s3")
    add("DVblindN", f"{META_V1}/blind.json", ["text", "n_eval"], "int")
    add("DZblindText", f"{META_V0}/blind.json", ["text", "auc"], "s3")
    add("DZblindImage", f"{META_V0}/blind.json", ["image", "auc"], "s3")
    man = load(f"{META_V1}/manifest.json")
    for split, n in man["n_groups"].items():
        name = "DVn" + "".join(w.capitalize() for w in split.split("_"))
        add_derived(name, n, f"{META_V1}/manifest.json", f"n_groups/{split}", "int")
    for split, W in (("train", "Train"), ("dev", "Dev"), ("calib", "Calib"), ("test_iid", "TestIid"),
                     ("test_heldout_template", "TestTemplate")):
        add(f"DVpart{W}", "configs/data_v1.json", ["orbit_partition", split], "s3" if split in ("dev", "calib") else "s2")
    add("SMevalCap", f"{SMOKE}/config.json", ["smoke", "eval_max_groups_per_split"], "int")
    sm = load(f"{SMOKE}/metrics.json")
    vals = [sm[f"{v}_seed0[content]"]["test_composition"]["overall"]["match"] for v in ("inbatch", "hardneg", "hardneg_eq")]
    add_derived("SMcontentMatchMin", min(vals), f"{SMOKE}/metrics.json",
                "min over variants of <variant>_seed0[content]/test_composition/overall/match")
    add_derived("SMcontentMatchMax", max(vals), f"{SMOKE}/metrics.json",
                "max over variants of <variant>_seed0[content]/test_composition/overall/match")


# ---------------------------------------------------------------------------
# Optional: split-semantics analysis and paired comparison
# ---------------------------------------------------------------------------

def latest(prefix):
    c = sorted(p for p in (ROOT / "runs").glob(prefix + "*") if (p / "manifest.json").exists())
    return str(c[-1].relative_to(ROOT)) if c else None


def split_semantics():
    d = latest("split_semantics_v1_")
    if not d:
        return False
    rel = f"{d}/split_semantics.json"
    s = load(rel)
    for split in ("train", "dev"):
        W = split.capitalize()
        add(f"SS{W}Attempts", rel, ["splits", split, "attempts"], "int")
        add(f"SS{W}Accept", rel, ["splits", split, "accept_rate"], "pct")
        add(f"SS{W}Orbits", rel, ["splits", split, "n_orbits_used"], "int")
        add(f"SS{W}OrbitsOwned", rel, ["splits", split, "n_orbits_owned"], "int")
        add(f"SS{W}ContentFilter", rel, ["splits", split, "content_edit_filter_keep_rate"], "pct")
    for panel in ("train", "dev"):
        W = panel.capitalize()
        add(f"SSpanel{W}N", rel, ["panel", panel, "n_groups"], "int")
        add(f"SSpanel{W}Orbits", rel, ["panel", panel, "n_orbits"], "int")
        add(f"SSpanel{W}BN", rel, ["panel", panel, "binding_necessary_fraction"], "pct")
    add("SSpanelDevExcluded", rel, ["panel", "dev", "excluded_previously_scored"], "int")
    add("SSpanelOverlap", rel, ["panel", "orbit_overlap_train_dev"], "int")
    add("SSorbitsTotal", rel, ["n_orbits_total"], "int")
    add("SSDevTVColor", rel, ["splits", "dev", "tv_to_train", "color"], "s2")
    add("SSDevTVShape", rel, ["splits", "dev", "tv_to_train", "shape"], "s2")
    add("SSHeldTemplateContentFilter", rel, ["splits", "test_heldout_template", "content_edit_filter_keep_rate"], "pct")
    add("SSCalibContentFilter", rel, ["splits", "calib", "content_edit_filter_keep_rate"], "pct")
    for op, W in (("replace_attr:color", "ReplColor"), ("replace_shape:-", "ReplShape"), ("swap_attr:color", "SwapColor")):
        add(f"SSDev{W}Kept", rel, ["splits", "dev", "orbit_filter_by_op", op, "kept"], "int")
        add(f"SSDev{W}Appl", rel, ["splits", "dev", "orbit_filter_by_op", op, "applicable"], "int")
    for col in ("green", "purple", "red", "blue", "yellow"):
        add(f"SSpanelDevColor{col.capitalize()}", rel, ["panel", "dev", "marginals", "color", col], "s2")
        add(f"SSpanelTrainColor{col.capitalize()}", rel, ["panel", "train", "marginals", "color", col], "s2")
    lines = []
    for split in ("train", "dev", "calib", "test_iid", "test_heldout_pairs", "test_composition", "test_heldout_template"):
        r = s["splits"][split]
        esc = {"train": "train", "dev": "dev", "calib": "calib", "test_iid": "test i.i.d.",
               "test_heldout_pairs": "test held-out pairs", "test_composition": "test composition",
               "test_heldout_template": "test held-out template"}[split]
        lines.append(f"{esc} & {r['n_groups']} & {r['attempts']} & {fmt(r['accept_rate'], 'pct')} & "
                     f"{r['n_orbits_owned']} & {r['n_orbits_used']} & {fmt(r['content_edit_filter_keep_rate'], 'pct')} & "
                     f"{fmt(r['binding_necessary_fraction'], 'pct')} \\\\")
    write_table("split_semantics.tex", lines, d)
    return True


def paired():
    d = latest("paired_v1_")
    if not d:
        return False
    rel = f"{d}/paired_summary.json"
    s = load(rel)
    add("PRtrainN", rel, ["panel", "train_groups"], "int")
    add("PRdevN", rel, ["panel", "dev_groups"], "int")
    add("PRdevBNN", rel, ["panel", "dev_binding_necessary"], "int")
    add("PRlambda", rel, ["config", "lam_eq"], "s2")
    add("PRsteps", rel, ["config", "steps"], "int")
    macros.append(("PRverdict", s["verdict"]["label"].replace("_", "\\_")))
    prov.append(("PRverdict", rel, "verdict/label", repr(s["verdict"]["label"]), "raw field"))
    for arm in ("A", "B"):
        for key, name in (("dev_bn_group", "DevBNGroup"), ("dev_bn_match", "DevBNMatch"), ("dev_bn_auc", "DevBNAUC"),
                          ("dev_all_group", "DevAllGroup"), ("train_bn_group", "TrainBNGroup"),
                          ("dev_binding_group", "DevBindGroup"), ("dev_relation_group", "DevRelGroup")):
            vals = [s["arms"][arm][sd][key] for sd in SEEDS]
            for sd, v in zip(SEEDS, vals):
                macros.append((f"PR{arm}{name}Seed{SEED_WORDS[sd]}", fmt(v, "s3")))
                prov.append((f"PR{arm}{name}Seed{SEED_WORDS[sd]}", rel, f"arms/{arm}/{sd}/{key}", repr(v), "raw field"))
            add_derived(f"PR{arm}{name}Mean", mean(vals), rel, f"mean over seeds of arms/{arm}/<seed>/{key}")
    for key, name in (("dev_bn_group", "Group"), ("dev_bn_match", "Match"), ("dev_bn_auc", "AUC")):
        add(f"PRdiff{name}", rel, ["paired", key, "mean_diff"])
        add(f"PRdiff{name}Lo", rel, ["paired", key, "ci95", 0])
        add(f"PRdiff{name}Hi", rel, ["paired", key, "ci95", 1])
        add(f"PRdiff{name}Pos", rel, ["paired", key, "n_seeds_positive"], "int")
    add("PRgradInitMean", rel, ["grad_ratio", "init_mean"], "s1")
    add("PRgradInitLamMean", rel, ["grad_ratio", "init_lam_scaled_mean"], "s2")
    add("PRzsBNGroup", rel, ["controls", "zero_shot_clip", "bn", "group"])
    add("PRzsBNMatch", rel, ["controls", "zero_shot_clip", "bn", "match"])
    add("PRzsAllGroup", rel, ["controls", "zero_shot_clip", "all", "group"])
    add("PRoracleBNGroup", rel, ["controls", "oracle", "bn", "group"])
    add("PRrandBNGroup", rel, ["controls", "random_mc300_bn", "group"])
    add("PRrandBNMatch", rel, ["controls", "random_mc300_bn", "match"])
    add("PRrenderImages", rel, ["renderer_audit", "images"], "int")
    add("PRrenderMismatch", rel, ["renderer_audit", "decode_mismatches"], "int")
    add("PRencCPU", rel, ["cost", "encoder_cpu_s"], "s1")
    for arm in ("A", "B"):
        for met, W in (("group", "Group"), ("match", "Match")):
            add(f"PR{arm}CI{W}Mean", rel, ["verdict", "arm_bn_ci", arm, met, "mean"])
            add(f"PR{arm}CI{W}Lo", rel, ["verdict", "arm_bn_ci", arm, met, "ci95", 0])
            add(f"PR{arm}CI{W}Hi", rel, ["verdict", "arm_bn_ci", arm, met, "ci95", 1])
        vals = []
        for sd in SEEDS:
            v = s["arms"][arm][sd]["loss_last"]
            vals.append(v)
            macros.append((f"PR{arm}LossLastSeed{SEED_WORDS[sd]}", fmt(v, "s2")))
            prov.append((f"PR{arm}LossLastSeed{SEED_WORDS[sd]}", rel, f"arms/{arm}/{sd}/loss_last", repr(v), "raw field"))
            c = s["arms"][arm][sd]["content_residual_bn"]
            macros.append((f"PR{arm}ResidSeed{SEED_WORDS[sd]}", fmt(c, "s2")))
            prov.append((f"PR{arm}ResidSeed{SEED_WORDS[sd]}", rel, f"arms/{arm}/{sd}/content_residual_bn", repr(c), "raw field"))
    det = load(f"{d}/arms_detail.json")
    gmax = {}
    for arm in ("A", "B"):
        for sd in SEEDS:
            v = max(h["grad_norm"] for h in det[arm][sd]["history"])
            gmax[(arm, sd)] = v
            add_derived(f"PR{arm}GradMaxSeed{SEED_WORDS[sd]}", v, f"{d}/arms_detail.json",
                        f"max of {arm}/{sd}/history[*]/grad_norm (logged every 100 steps)", "s1")
    add_derived("PRgradMaxAll", max(gmax.values()), f"{d}/arms_detail.json",
                "max over arms and seeds of logged grad_norm", "s1")
    add_derived("PRgradMinOfMax", min(gmax.values()), f"{d}/arms_detail.json",
                "min over arms and seeds of the max logged grad_norm", "s1")
    tr = load(f"{d}/config.json")["paired"]
    add("PRlr", f"{d}/config.json", ["paired", "train", "lr"], "s2")
    add("PRbatch", f"{d}/config.json", ["paired", "train", "batch_groups"], "int")
    add("PRwd", f"{d}/config.json", ["paired", "train", "weight_decay"], "e2")
    add("PRcpuTotal", rel, ["cost", "stage_cpu_s"], "s1")
    add("PRpeakRSS", rel, ["cost", "peak_rss_mib"], "s1")
    lines = []
    for label, key in (("Dev \\bn{} group score", "dev_bn_group"), ("Dev \\bn{} GroupMatch", "dev_bn_match"),
                       ("Dev \\bn{} pair AUC", "dev_bn_auc"), ("Dev binding-only group", "dev_binding_group"),
                       ("Dev relation-only group", "dev_relation_group"), ("Dev all-kinds group", "dev_all_group"),
                       ("Dev all-kinds GroupMatch", "dev_all_match"), ("Train \\bn{} group (fit)", "train_bn_group"),
                       ("Final task loss", "loss_last")):
        def m(v):
            if key == "loss_last":
                return f"{v:.2f}".replace("-", "$-$")
            return ("$-$" + fmt(-v, "c3")) if v < 0 else fmt(v, "c3")
        a = "/".join(m(s["arms"]["A"][sd][key]) for sd in SEEDS)
        b = "/".join(m(s["arms"]["B"][sd][key]) for sd in SEEDS)
        diff = "/".join(m(s["arms"]["B"][sd][key] - s["arms"]["A"][sd][key]) for sd in SEEDS)
        lines.append(f"{label} & {a} & {b} & {diff} \\\\")
    write_table("paired.tex", lines, d)
    return True


HEADERS = {
    "small_panel.tex": ("lcccc", "Scorer & \\bn{} group & \\bn{} GroupMatch & \\bn{} pair AUC & all: group"),
    "small_panel_all.tex": ("lccc", "Scorer & all: group & all: GroupMatch & all: pair AUC"),
    "split_semantics.tex": ("lrrrrrrr", "Split & groups & attempts & accept & orbits & used & keep & \\bn{}"),
    "paired.tex": ("lccc", "Seeds 0/1/2 & A: hard-neg. & B: + edit-cons. & B$-$A"),
}


def write_table(name, lines, source):
    """Write a complete tabular (rules included) so the .tex never \\input's inside a tabular."""
    out = ROOT / "paper" / "tables" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    spec, header = HEADERS[name]
    body = [f"% GENERATED by scripts/export_paper_numbers.py from {source}; do not edit by hand.",
            f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header + " \\\\", "\\midrule", *lines,
            "\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(body) + "\n")


def main():
    small_panel()
    data_audits()
    have_ss = split_semantics()
    have_pr = paired()
    gen = ROOT / "paper" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    body = ["% GENERATED by scripts/export_paper_numbers.py; do not edit by hand.",
            f"\\newif\\ifhavesplitsemantics\\havesplitsemantics{'true' if have_ss else 'false'}",
            f"\\newif\\ifhavepaired\\havepaired{'true' if have_pr else 'false'}"]
    seen = set()
    for name, val in macros:
        if name in seen:
            sys.exit(f"duplicate macro {name}")
        if not name.isalpha():
            sys.exit(f"macro name must be letters only: {name}")
        seen.add(name)
        body.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    (gen / "numbers.tex").write_text("\n".join(body) + "\n")
    (gen / "provenance.tsv").write_text("macro\tsource_file\tjson_field\tvalue\trule\n"
                                        + "\n".join("\t".join(map(str, r)) for r in prov) + "\n")
    print(f"{len(macros)} macros; split_semantics={have_ss}; paired={have_pr}")


if __name__ == "__main__":
    main()
