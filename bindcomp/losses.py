"""Losses on embeddings with explicit gradients.

info_nce       symmetric InfoNCE with diagonal targets and an ``allowed`` mask
               selecting which off-diagonal pairs act as negatives. The
               hard-negative baseline allows the group partner; the in-batch
               baseline masks it out. Pairs the oracle marks true are never
               negatives (false-negative masking uses the same metadata for
               every variant).
edit_consistency
               symmetric InfoNCE over cosine similarities of per-group edit
               vectors  dI_g = e(I_g1) - e(I_g0)  and  dT_g = e(T_g1) - e(T_g0):
               the change an edit induces in the image embedding must identify
               the change the same edit induces in the text embedding, against
               other groups' edits. Its trivial minimizer is NOT the zero
               solution (cos of a near-zero vector gives uniform logits and a
               loss of log B), unlike delta_mse below.
delta_mse      || dI_g - dT_g ||^2 averaged over groups. Kept only to document
               the zero/constant-solution failure mode; not used in training.

Identity worth keeping in mind: for a 2x2 group,
  s00 + s11 - s01 - s10 = <e(I0) - e(I1), e(T0) - e(T1)> = <dI, dT>,
so the within-group part of edit alignment is exactly what the hard-negative
loss and the GroupMatch metric (Zhu et al., 2025) already measure; the only
new signal in ``edit_consistency`` is the *across-group* contrast of edits.
"""

from __future__ import annotations

from math import exp, log, sqrt


def _logsumexp(xs):
    m = max(xs)
    return m + log(sum(exp(x - m) for x in xs))


def info_nce(S, allowed, scale=1.0):
    """S[i][j]: score of image i with text j; target j == i.

    Returns (loss, dS) with loss = 0.5 * (mean row CE + mean column CE).
    """
    N = len(S)
    dS = [[0.0] * N for _ in range(N)]
    loss = 0.0
    for i in range(N):  # image -> text
        js = [j for j in range(N) if j == i or allowed[i][j]]
        logits = [scale * S[i][j] for j in js]
        lse = _logsumexp(logits)
        loss += 0.5 * (lse - scale * S[i][i]) / N
        for j, z in zip(js, logits):
            dS[i][j] += 0.5 * scale * (exp(z - lse) - (1.0 if j == i else 0.0)) / N
    for j in range(N):  # text -> image
        is_ = [i for i in range(N) if i == j or allowed[i][j]]
        logits = [scale * S[i][j] for i in is_]
        lse = _logsumexp(logits)
        loss += 0.5 * (lse - scale * S[j][j]) / N
        for i, z in zip(is_, logits):
            dS[i][j] += 0.5 * scale * (exp(z - lse) - (1.0 if i == j else 0.0)) / N
    return loss, dS


def edit_consistency(DI, DT, tau=0.1, eps=1e-8, allowed=None):
    """Cosine-InfoNCE between image edit vectors DI[g] and text edit vectors DT[h]."""
    B = len(DI)
    nI = [sqrt(sum(x * x for x in v) + eps) for v in DI]
    nT = [sqrt(sum(x * x for x in v) + eps) for v in DT]
    C = [[sum(a * b for a, b in zip(DI[g], DT[h])) / (nI[g] * nT[h]) for h in range(B)] for g in range(B)]
    ok = allowed or [[True] * B for _ in range(B)]
    G = [[0.0] * B for _ in range(B)]
    loss = 0.0
    for g in range(B):
        hs = [h for h in range(B) if h == g or ok[g][h]]
        z = [C[g][h] / tau for h in hs]
        lse = _logsumexp(z)
        loss += 0.5 * (lse - C[g][g] / tau) / B
        for h, zz in zip(hs, z):
            G[g][h] += 0.5 * (exp(zz - lse) - (1.0 if h == g else 0.0)) / (B * tau)
    for h in range(B):
        gs = [g for g in range(B) if g == h or ok[g][h]]
        z = [C[g][h] / tau for g in gs]
        lse = _logsumexp(z)
        loss += 0.5 * (lse - C[h][h] / tau) / B
        for g, zz in zip(gs, z):
            G[g][h] += 0.5 * (exp(zz - lse) - (1.0 if g == h else 0.0)) / (B * tau)
    d = len(DI[0])
    dDI = [[0.0] * d for _ in range(B)]
    dDT = [[0.0] * d for _ in range(B)]
    for g in range(B):
        for h in range(B):
            w = G[g][h]
            if w == 0.0:
                continue
            inv = 1.0 / (nI[g] * nT[h])
            cI = C[g][h] / (nI[g] * nI[g])
            cT = C[g][h] / (nT[h] * nT[h])
            for k in range(d):
                dDI[g][k] += w * (DT[h][k] * inv - cI * DI[g][k])
                dDT[h][k] += w * (DI[g][k] * inv - cT * DT[h][k])
    return loss, dDI, dDT, C


def delta_mse(DI, DT):
    B = len(DI)
    loss = 0.0
    dDI, dDT = [], []
    for a, b in zip(DI, DT):
        diff = [x - y for x, y in zip(a, b)]
        loss += sum(x * x for x in diff) / B
        dDI.append([2 * x / B for x in diff])
        dDT.append([-2 * x / B for x in diff])
    return loss, dDI, dDT
