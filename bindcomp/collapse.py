"""Detectors for degenerate representations.

Flags (thresholds are fixed here, not tuned to make tests pass):
  ZERO                 mean embedding norm below ``zero_tol``
  CONSTANT             embeddings vary negligibly across inputs relative to their norm
  LOW_RANK             participation ratio (tr C)^2 / tr(C^2) < ``pr_min``   (info)
  BINDING_INSENSITIVE  binding-only edits barely move the binding part
  CONTENT_LEAK         binding-only edits move the content part (should be ~0 by
                       construction when render nuisance is held fixed)
"""

from __future__ import annotations

from math import sqrt
from statistics import median

ZERO_TOL = 1e-8
CONST_REL_TOL = 1e-6
PR_MIN = 1.5
BIND_SENS_MIN = 1e-3


def _norm(v):
    return sqrt(sum(x * x for x in v))


def embedding_stats(vectors):
    n = len(vectors)
    d = len(vectors[0])
    mean = [sum(v[i] for v in vectors) / n for i in range(d)]
    centered = [[v[i] - mean[i] for i in range(d)] for v in vectors]
    var = [sum(c[i] ** 2 for c in centered) / n for i in range(d)]
    cov = [[sum(c[i] * c[j] for c in centered) / n for j in range(d)] for i in range(d)]
    tr = sum(var)
    tr2 = sum(cov[i][j] ** 2 for i in range(d) for j in range(d))
    mean_norm = sum(_norm(v) for v in vectors) / n
    mean_std = sum(sqrt(x) for x in var) / d
    return {
        "n": n,
        "dim": d,
        "mean_norm": mean_norm,
        "mean_std": mean_std,
        "rel_std": mean_std / (mean_norm + 1e-30),
        "participation_ratio": (tr * tr / tr2) if tr2 > 0 else 0.0,
    }


def collapse_flags(stats):
    flags = []
    if stats["mean_norm"] < ZERO_TOL:
        flags.append("ZERO")
    elif stats["rel_std"] < CONST_REL_TOL:
        flags.append("CONSTANT")
    if stats["participation_ratio"] < PR_MIN:
        flags.append("LOW_RANK")
    return flags


def edit_sensitivity(pairs):
    """pairs: list of (before, after) vectors. Returns median ||after-before|| / median ||before||."""
    if not pairs:
        return None
    deltas = [_norm([a - b for a, b in zip(y, x)]) for x, y in pairs]
    base = median(_norm(x) for x, _ in pairs)
    return median(deltas) / (base + 1e-30)


def head_report(head, enc_groups, hold_nuisance=True):
    """Collapse / sensitivity report of a head on encoded groups.

    With ``hold_nuisance`` both members of a group are encoded from the same
    render seed so that differences come only from the edit.
    """
    from .encode import image_tokens_for

    c_img, b_img, c_txt, b_txt = [], [], [], []
    bind_img, bind_txt, cont_img = [], [], []
    for g in enc_groups:
        embs_i, embs_t = [], []
        for m in (0, 1):
            toks = image_tokens_for(g, m, same_seed=hold_nuisance)
            ci, bi, _ = head.encode_image(toks)
            ct, bt, _ = head.encode_text(g.txt[m])
            embs_i.append((ci, bi))
            embs_t.append((ct, bt))
            c_img.append(ci)
            b_img.append(bi)
            c_txt.append(ct)
            b_txt.append(bt)
        if set(g.kinds) <= {"binding", "relation"}:
            bind_img.append((embs_i[0][1], embs_i[1][1]))
            bind_txt.append((embs_t[0][1], embs_t[1][1]))
            cont_img.append((embs_i[0][0], embs_i[1][0]))
    rep = {}
    for name, vs in (("c_img", c_img), ("b_img", b_img), ("c_txt", c_txt), ("b_txt", b_txt)):
        st = embedding_stats(vs)
        st["flags"] = collapse_flags(st)
        rep[name] = st
    rep["binding_sensitivity_img"] = edit_sensitivity(bind_img)
    rep["binding_sensitivity_txt"] = edit_sensitivity(bind_txt)
    rep["content_residual_img"] = (
        max(_norm([a - b for a, b in zip(y, x)]) for x, y in cont_img) if cont_img else None
    )
    flags = []
    for name in ("c_img", "b_img", "c_txt", "b_txt"):
        flags += [f"{name}:{f}" for f in rep[name]["flags"]]
    for side in ("img", "txt"):
        s = rep[f"binding_sensitivity_{side}"]
        if s is not None and s < BIND_SENS_MIN:
            flags.append(f"b_{side}:BINDING_INSENSITIVE")
    if hold_nuisance and rep["content_residual_img"] is not None and rep["content_residual_img"] > 1e-9:
        flags.append("c_img:CONTENT_LEAK")
    rep["flags"] = flags
    return rep
