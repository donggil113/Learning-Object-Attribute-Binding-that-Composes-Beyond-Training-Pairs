"""Training of the shared head under different losses.

Variants (identical head, data, batches, optimizer and false-negative masking;
they differ ONLY in the loss):
  inbatch     InfoNCE; the group partner is masked out of the negatives.
  hardneg     InfoNCE; the group partner is a negative (same-data hard-negative
              baseline, NegCLIP / CounterCurate "grouping" / VisMin-CLIP style).
  hardneg_eq  hardneg + lam_eq * edit_consistency over the batch's groups.

Information access: all three see the same groups (both members), the same
oracle truth table for false-negative masking, and the same group pairing.
``inbatch`` does not use the pairing as a contrast; ``hardneg_eq`` uses the
pairing for both the hard negative and the edit vectors. No variant sees the
op kind, the base/edited flag, or held-out metadata.
"""

from __future__ import annotations

import random
import time

from .losses import edit_consistency, info_nce
from .optim import Adam, grad_norm
from .oracle import satisfies

VARIANTS = {
    "inbatch": {"hard_neg": False, "lam_eq": 0.0},
    "hardneg": {"hard_neg": True, "lam_eq": 0.0},
    "hardneg_eq": {"hard_neg": True, "lam_eq": None},  # lam_eq from config
}


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def batch_forward(head, batch, hard_neg, lam_eq, scale=1.0, tau_eq=0.1, want_grads=True):
    """Loss and gradients for a batch of EncGroups (both members of each group)."""
    items = [(g, m) for g in batch for m in (0, 1)]
    N = len(items)
    enc_i = [head.encode_image(g.img[m]) for g, m in items]
    enc_t = [head.encode_text(g.txt[m]) for g, m in items]
    eI = [c + b for c, b, _ in enc_i]
    eT = [c + b for c, b, _ in enc_t]
    S = [[_dot(eI[i], eT[j]) for j in range(N)] for i in range(N)]
    allowed = [[False] * N for _ in range(N)]
    n_fn = 0
    for i, (gi, mi) in enumerate(items):
        for j, (gj, mj) in enumerate(items):
            if i == j:
                continue
            if satisfies(gi.scenes[mi], gj.descs[mj]):
                n_fn += 1  # oracle-true off-diagonal pair: never a negative
                continue
            if gi is gj and not hard_neg:
                continue
            allowed[i][j] = True
    l_task, dS = info_nce(S, allowed, scale)
    out = {"loss": l_task, "task": l_task, "eq": 0.0, "n_false_neg_masked": n_fn}
    dEI = [[sum(dS[i][j] * eT[j][k] for j in range(N)) for k in range(len(eT[0]))] for i in range(N)]
    dET = [[sum(dS[i][j] * eI[i][k] for i in range(N)) for k in range(len(eI[0]))] for j in range(N)]
    if lam_eq:
        B = len(batch)
        DI = [[a - b for a, b in zip(eI[2 * g + 1], eI[2 * g])] for g in range(B)]
        DT = [[a - b for a, b in zip(eT[2 * g + 1], eT[2 * g])] for g in range(B)]
        l_eq, dDI, dDT, _ = edit_consistency(DI, DT, tau=tau_eq)
        out["eq"] = l_eq
        out["loss"] = l_task + lam_eq * l_eq
        for g in range(B):
            for k in range(len(DI[0])):
                dEI[2 * g + 1][k] += lam_eq * dDI[g][k]
                dEI[2 * g][k] -= lam_eq * dDI[g][k]
                dET[2 * g + 1][k] += lam_eq * dDT[g][k]
                dET[2 * g][k] -= lam_eq * dDT[g][k]
    if not want_grads:
        return out, None
    grads = head.zero_grads()
    hc = head.hc
    for i in range(N):
        head.backward(enc_i[i][2], dEI[i][:hc], dEI[i][hc:], grads)
        head.backward(enc_t[i][2], dET[i][:hc], dET[i][hc:], grads)
    return out, grads


def train(head, train_groups, tcfg, variant, seed, log=None):
    spec = dict(VARIANTS[variant])
    if spec["lam_eq"] is None:
        spec["lam_eq"] = tcfg["lam_eq"]
    rng = random.Random(f"train-order:{seed}")
    opt = Adam(head.p, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    history = []
    order = []
    t0 = time.perf_counter()
    for step in range(1, tcfg["steps"] + 1):
        batch = []
        while len(batch) < tcfg["batch_groups"]:
            if not order:
                order = list(range(len(train_groups)))
                rng.shuffle(order)
            batch.append(train_groups[order.pop()])
        out, grads = batch_forward(head, batch, spec["hard_neg"], spec["lam_eq"],
                                   scale=tcfg["logit_scale"], tau_eq=tcfg["tau_eq"])
        out["grad_norm"] = grad_norm(grads)
        opt.step(grads)
        out["step"] = step
        out["elapsed_s"] = round(time.perf_counter() - t0, 3)
        history.append(out)
        if log and (step == 1 or step % tcfg["log_every"] == 0 or step == tcfg["steps"]):
            log(f"[{variant} seed={seed}] step {step} loss={out['loss']:.4f} task={out['task']:.4f} "
                f"eq={out['eq']:.4f} |g|={out['grad_norm']:.4f} t={out['elapsed_s']}s")
    return history
