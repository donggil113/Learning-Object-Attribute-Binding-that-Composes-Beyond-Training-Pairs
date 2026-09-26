"""Torch bridge of the existing head and losses (no new architecture or loss).

``TorchFactorizedHead`` copies the parameters of ``head.FactorizedHead`` (same
seed -> same initialization) and computes the same content/binding outputs on
batched, padded token tensors. ``info_nce`` / ``edit_consistency`` mirror
``losses.py``. Equality of outputs, losses and gradients with the reference
implementation is tested in tests/test_pixel_path.py. Autograd is torch's; the
only reason for the bridge is speed at d=512.
"""

from __future__ import annotations

import random
import time

import torch

from .head import FactorizedHead
from .oracle import satisfies

DTYPE = torch.float64


class TorchFactorizedHead(torch.nn.Module):
    def __init__(self, d, hc=8, hb=8, window=5, seed=0, init_scale=1.0):
        super().__init__()
        ref = FactorizedHead(d=d, hc=hc, hb=hb, window=window, seed=seed, init_scale=init_scale)
        self.config = dict(ref.config)
        self.hc, self.hb, self.offsets = hc, hb, list(ref.offsets)
        self.p = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(v, dtype=DTYPE)) for k, v in ref.p.items()})

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def encode_image(self, toks):
        """toks: (N, n, d) -> c (N, hc), b (N, hb)."""
        c = toks.mean(dim=1) @ self.p["Wc_img"].T
        P = toks @ self.p["A_img"].T
        Q = toks @ self.p["A2_img"].T
        return c, (P * Q).mean(dim=1)

    def encode_text(self, toks, lengths):
        """toks: (N, T, d) zero-padded; lengths: (N,) -> c (N, hc), b (N, hb)."""
        L = lengths.to(DTYPE)[:, None]
        c = (toks.sum(dim=1) / L) @ self.p["Wc_txt"].T
        P = toks @ self.p["B_txt"].T  # padded rows are exactly zero (no bias)
        Q = toks @ self.p["B2_txt"].T
        T = toks.shape[1]
        b = torch.zeros(toks.shape[0], self.hb, dtype=DTYPE)
        for a, off in zip(self.p["alpha"], self.offsets):
            if abs(off) >= T:
                continue
            if off > 0:
                b = b + a * (P[:, : T - off] * Q[:, off:]).sum(dim=1)
            else:
                b = b + a * (P[:, -off:] * Q[:, : T + off]).sum(dim=1)
        return c, b / L


def pad(seqs):
    lengths = torch.tensor([s.shape[0] for s in seqs])
    out = torch.zeros(len(seqs), int(lengths.max()), seqs[0].shape[1], dtype=DTYPE)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out, lengths


def info_nce(S, allowed, scale=1.0):
    """Mirror of losses.info_nce: diagonal targets, ``allowed`` off-diagonal negatives."""
    N = S.shape[0]
    keep = allowed | torch.eye(N, dtype=torch.bool)
    z = (scale * S).masked_fill(~keep, float("-inf"))
    tgt = torch.arange(N)
    return 0.5 * (torch.nn.functional.cross_entropy(z, tgt) + torch.nn.functional.cross_entropy(z.T, tgt))


def edit_consistency(DI, DT, tau=0.1, eps=1e-8):
    """Mirror of losses.edit_consistency (cosine-InfoNCE over per-group edit vectors)."""
    nI = torch.sqrt((DI * DI).sum(dim=1) + eps)
    nT = torch.sqrt((DT * DT).sum(dim=1) + eps)
    C = (DI @ DT.T) / (nI[:, None] * nT[None, :])
    tgt = torch.arange(DI.shape[0])
    return 0.5 * (torch.nn.functional.cross_entropy(C / tau, tgt) + torch.nn.functional.cross_entropy(C.T / tau, tgt))


def batch_embeddings(head, batch):
    """batch: list of feature groups with .img (2, n, d) and .txt (list of 2 (L, d))."""
    imgs = torch.cat([g.img for g in batch]).to(DTYPE)
    txt, lengths = pad([t.to(DTYPE) for g in batch for t in g.txt])
    ci, bi = head.encode_image(imgs)
    ct, bt = head.encode_text(txt, lengths)
    return torch.cat([ci, bi], dim=1), torch.cat([ct, bt], dim=1)


def negative_mask(batch, hard_neg):
    """Same rule as train.batch_forward: oracle-true pairs never negatives; partner per variant.

    Uses scene/caption metadata as supervision (labels), never as model input.
    """
    items = [(g, m) for g in batch for m in (0, 1)]
    N = len(items)
    allowed = torch.zeros(N, N, dtype=torch.bool)
    for i, (gi, mi) in enumerate(items):
        for j, (gj, mj) in enumerate(items):
            if i == j or satisfies(gi.scenes[mi], gj.descs[mj]):
                continue
            if gi is gj and not hard_neg:
                continue
            allowed[i, j] = True
    return allowed


def batch_loss(head, batch, hard_neg, lam_eq, scale=1.0, tau_eq=0.1):
    eI, eT = batch_embeddings(head, batch)
    l_task = info_nce(eI @ eT.T, negative_mask(batch, hard_neg), scale)
    l_eq = torch.zeros((), dtype=DTYPE)
    if lam_eq:
        l_eq = edit_consistency(eI[1::2] - eI[0::2], eT[1::2] - eT[0::2], tau_eq)
    return l_task + lam_eq * l_eq, l_task, l_eq


def grad_norm_of(head, loss):
    grads = torch.autograd.grad(loss, list(head.parameters()), retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g * g).sum() for g in grads if g is not None)))


def train(head, groups, tcfg, hard_neg, seed, log=None):
    """Same batch-order rule and optimizer semantics as train.train (Adam, decoupled decay)."""
    rng = random.Random(f"train-order:{seed}")
    opt = torch.optim.AdamW(head.parameters(), lr=tcfg["lr"], betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=tcfg["weight_decay"])
    hist, order = [], []
    t0 = time.perf_counter()
    for step in range(1, tcfg["steps"] + 1):
        batch = []
        while len(batch) < min(tcfg["batch_groups"], len(groups)):
            if not order:
                order = list(range(len(groups)))
                rng.shuffle(order)
            batch.append(groups[order.pop()])
        loss, l_task, _ = batch_loss(head, batch, hard_neg, 0.0, tcfg["logit_scale"])
        opt.zero_grad()
        loss.backward()
        gn = float(torch.sqrt(sum((p.grad * p.grad).sum() for p in head.parameters())))
        opt.step()
        hist.append({"step": step, "task": l_task.item(), "grad_norm": gn,
                     "elapsed_s": round(time.perf_counter() - t0, 3)})
        if log and (step == 1 or step % tcfg["log_every"] == 0 or step == tcfg["steps"]):
            log(f"  step {step} task={l_task.item():.4f} |g|={gn:.4f}")
    return hist


class TorchHeadScorer:
    """Scores a feature group using feature tensors only (never ``meta``)."""

    def __init__(self, head, channels="all"):
        self.head, self.channels = head, channels
        self.name = f"pixel_head[{channels}]"
        self.provenance = "pixels + caption text"

    def _pick(self, c, b):
        return {"all": torch.cat([c, b], 1), "content": c, "binding": b}[self.channels]

    @torch.no_grad()
    def score_group(self, fg):
        eI = self._pick(*self.head.encode_image(fg.img.to(DTYPE)))
        txt, lengths = pad([t.to(DTYPE) for t in list(fg.txt) + list(fg.txt_hp)])
        eT = self._pick(*self.head.encode_text(txt, lengths))
        S = (eI @ eT[:2].T).tolist()
        return S, [float(eI[i] @ eT[2 + i]) for i in (0, 1)]
