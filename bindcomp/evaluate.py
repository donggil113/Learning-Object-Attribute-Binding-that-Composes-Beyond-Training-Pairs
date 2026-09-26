"""Pair-level and group-level evaluation shared by every scorer.

Every scorer implements ``score_group(eg) -> (S, S_hp)`` where S[i][t] is the
score of image member i with caption member t, and S_hp[i] is the score of
image i with the meaning-preserving paraphrase of its own caption.

Group metrics (strict inequalities with a 1e-9 relative tie tolerance; ties fail):
  text   S00 > S01 and S11 > S10            (Winoground text score)
  image  S00 > S10 and S11 > S01            (Winoground image score)
  group  text and image                     (Winoground group score)
  match  S00 + S11 > S01 + S10              (GroupMatch, Zhu et al. 2025, k=2)
  aug    for each image i: S_ii > S_i,1-i and S_hp_i > S_i,1-i
         (augmented accuracy with hard positives, Kamath et al. 2024)
Chance for i.i.d. continuous scores: text 1/4, image 1/4, group 1/6, match 1/2,
aug 1/3 (per image, own caption and paraphrase both beat the other caption).

Pair metrics: the four (image, caption) pairs of every group, labeled by the
oracle; AUC (threshold-free) and accuracy at a threshold fitted on the
calibration split only.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .oracle import satisfies


TIE_RTOL = 1e-9


def gt(a, b):
    """Strictly greater beyond float round-off.

    Scores that are mathematically equal (e.g. a bag-of-words channel on two
    captions with the same words, summed in a different order) can differ in
    the last bits; a plain ``>`` would turn that noise into coin-flip wins.
    """
    return a - b > TIE_RTOL * max(1.0, abs(a), abs(b))


def group_outcome(S, S_hp=None):
    text = gt(S[0][0], S[0][1]) and gt(S[1][1], S[1][0])
    image = gt(S[0][0], S[1][0]) and gt(S[1][1], S[0][1])
    res = {
        "text": float(text),
        "image": float(image),
        "group": float(text and image),
        "match": float(gt(S[0][0] + S[1][1], S[0][1] + S[1][0])),
    }
    if S_hp is not None:
        ok = [gt(S[i][i], S[i][1 - i]) and gt(S_hp[i], S[i][1 - i]) for i in (0, 1)]
        # Per-image (Kamath-style) accuracy is NOT blind-proof: a text-only
        # scorer wins it for one of the two images whenever it prefers the right
        # caption type. ``aug_group`` requires both images and is blind-proof.
        res["aug"] = sum(map(float, ok)) / 2
        res["aug_group"] = float(all(ok))
    return res


def evaluate_groups(scorer, enc_groups):
    rows = []
    for eg in enc_groups:
        S, S_hp = scorer.score_group(eg)
        r = group_outcome(S, S_hp)
        r.update(gid=eg.gid, kind=eg.kind_label, same_words=eg.same_words, S=S, S_hp=S_hp)
        rows.append(r)
    return rows


METRICS = ("text", "image", "group", "match", "aug", "aug_group")


def tie_rate(rows):
    """Fraction of groups where any compared pair of scores ties (same tolerance as ``gt``)."""
    def tied(a, b):
        return not gt(a, b) and not gt(b, a)

    n = 0
    for r in rows:
        S = r["S"]
        pairs = [(S[0][0], S[0][1]), (S[1][1], S[1][0]), (S[0][0], S[1][0]), (S[1][1], S[0][1]),
                 (S[0][0] + S[1][1], S[0][1] + S[1][0])]
        n += any(tied(a, b) for a, b in pairs)
    return n / len(rows) if rows else None


def summarize(rows, key=None):
    if key is None:
        buckets = {"all": rows}
    else:
        buckets = defaultdict(list)
        for r in rows:
            buckets[key(r)].append(r)
    out = {}
    for b, rs in sorted(buckets.items()):
        out[b] = {"n": len(rs)}
        for m in METRICS:
            vals = [r[m] for r in rs if m in r]
            if vals:
                out[b][m] = sum(vals) / len(vals)
    return out


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    rng = random.Random(f"bootstrap:{seed}")
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def pair_scores(rows, enc_groups):
    """Flatten the four pairs of each group with oracle labels."""
    scores, labels = [], []
    for r, eg in zip(rows, enc_groups):
        for i in (0, 1):
            for t in (0, 1):
                scores.append(r["S"][i][t])
                labels.append(int(satisfies(eg.scenes[i], eg.descs[t])))
    return scores, labels


def auc(scores, labels):
    """Mann-Whitney AUC with average ranks for ties."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    r_pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auc_bootstrap_ci(scores, labels, n_boot=200, alpha=0.05, seed=0):
    rng = random.Random(f"auc-bootstrap:{seed}")
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a = auc([scores[i] for i in idx], [labels[i] for i in idx])
        if a is not None:
            vals.append(a)
    vals.sort()
    return vals[int((alpha / 2) * len(vals))], vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]


def fit_threshold(scores, labels):
    """Accuracy-maximizing threshold (predict positive iff score > thr)."""
    cand = sorted(set(scores))
    mids = [cand[0] - 1.0] + [(a + b) / 2 for a, b in zip(cand, cand[1:])] + [cand[-1] + 1.0]
    best, best_acc = mids[0], -1.0
    for thr in mids:
        acc = sum((s > thr) == bool(y) for s, y in zip(scores, labels)) / len(labels)
        if acc > best_acc:
            best, best_acc = thr, acc
    return best, best_acc


def pair_accuracy(scores, labels, thr):
    return sum((s > thr) == bool(y) for s, y in zip(scores, labels)) / len(labels)


def binding_necessary(kind_label):
    return set(kind_label.split("+")) <= {"binding", "relation"}


def full_report(scorer, splits_enc, calib_key="calib", n_boot=1000):
    """Group + pair metrics for every split; threshold fitted on ``calib_key`` only."""
    rep = {}
    rows_by = {k: evaluate_groups(scorer, v) for k, v in splits_enc.items()}
    thr = None
    if calib_key in rows_by:
        s, l = pair_scores(rows_by[calib_key], splits_enc[calib_key])
        thr, _ = fit_threshold(s, l)
    for k, rows in rows_by.items():
        s, l = pair_scores(rows, splits_enc[k])
        rep[k] = {
            "overall": summarize(rows)["all"],
            "by_kind": summarize(rows, key=lambda r: r["kind"]),
            # Groups whose edits leave object/attribute multisets unchanged: the
            # content channel cannot separate their images, so these are the
            # groups where a gain can be attributed to binding.
            "binding_necessary": summarize(
                [r for r in rows if binding_necessary(r["kind"])]
            ).get("all"),
            "binding_relation_same_words": summarize(
                [r for r in rows if r["same_words"] and binding_necessary(r["kind"])]
            ).get("all"),
            "group_ci95": bootstrap_ci([r["group"] for r in rows], n_boot=n_boot),
            "match_ci95": bootstrap_ci([r["match"] for r in rows], n_boot=n_boot),
            "pair_auc": auc(s, l),
            "pair_acc_at_calib_threshold": None if thr is None else pair_accuracy(s, l, thr),
        }
    rep["calib_threshold"] = thr
    return rep
