"""Small content/binding head on top of frozen token features (manual gradients).

For a token set / sequence the head outputs two parts:

  content c : linear projection of the MEAN token. Mean pooling of additive
              tokens is exactly invariant to re-assigning attributes between
              objects (image) and to re-ordering words (text), so ``c`` cannot
              see binding by construction (tested, not learned).
  binding b : multiplicative (Hadamard) interaction pooled over tokens.
              image: b = 1/n sum_k (A v_k) * (A2 v_k)            (within-object)
              text : b = 1/L sum_t sum_d alpha_d (B u_t) * (B2 u_{t+d})
                     over relative offsets d in [-W, W] \\ {0}   (within-window)

The score is s(I, T) = <c_I, c_T> + <b_I, b_T> = <[c_I; b_I], [c_T; b_T]>.
All variants in this repo share this head; they differ only in the loss.

Note (architectural, not learned): the image binding part is additive over
objects, so for edits touching disjoint objects the change in b composes
exactly: b(e2(e1(s))) - b(s) = [b(e1(s)) - b(s)] + [b(e2(s)) - b(s)] when the
render nuisance is held fixed. Held-out-composition results must be read with
this in mind (see tests/test_head.py::test_image_binding_additive_over_disjoint_edits).
"""

from __future__ import annotations

import hashlib
import json
import random
from math import sqrt

PARAM_NAMES = ("Wc_img", "A_img", "A2_img", "Wc_txt", "B_txt", "B2_txt", "alpha")


def _mat(rng, r, c, std):
    return [[rng.gauss(0.0, std) for _ in range(c)] for _ in range(r)]


def _matvec(M, x):
    return [sum(m * v for m, v in zip(row, x)) for row in M]


def _mean(vs):
    n = len(vs)
    return [sum(col) / n for col in zip(*vs)]


def _add_outer(G, a, x):
    for i, ai in enumerate(a):
        if ai != 0.0:
            row = G[i]
            for j, xj in enumerate(x):
                row[j] += ai * xj


class FactorizedHead:
    def __init__(self, d, hc=8, hb=8, window=5, seed=0, init_scale=1.0):
        self.d, self.hc, self.hb, self.window = d, hc, hb, window
        self.offsets = [o for o in range(-window, window + 1) if o != 0]
        rng = random.Random(f"head:{seed}")
        std = init_scale / sqrt(d)
        self.p = {
            "Wc_img": _mat(rng, hc, d, std),
            "A_img": _mat(rng, hb, d, std),
            "A2_img": _mat(rng, hb, d, std),
            "Wc_txt": _mat(rng, hc, d, std),
            "B_txt": _mat(rng, hb, d, std),
            "B2_txt": _mat(rng, hb, d, std),
            # Uninformative start: equal weight on every offset.
            "alpha": [1.0 / len(self.offsets)] * len(self.offsets),
        }
        self.config = dict(d=d, hc=hc, hb=hb, window=window, seed=seed, init_scale=init_scale)

    # ---- parameters ---------------------------------------------------------
    def zero_grads(self):
        g = {}
        for k, v in self.p.items():
            g[k] = [[0.0] * len(r) for r in v] if isinstance(v[0], list) else [0.0] * len(v)
        return g

    def n_params(self):
        return sum(len(v) * len(v[0]) if isinstance(v[0], list) else len(v) for v in self.p.values())

    def param_hash(self):
        blob = json.dumps({k: self.p[k] for k in PARAM_NAMES}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def state(self):
        return {"config": self.config, "params": self.p}

    # ---- image --------------------------------------------------------------
    def encode_image(self, toks):
        n = len(toks)
        m = _mean(toks)
        c = _matvec(self.p["Wc_img"], m)
        P = [_matvec(self.p["A_img"], v) for v in toks]
        Q = [_matvec(self.p["A2_img"], v) for v in toks]
        b = [sum(P[k][i] * Q[k][i] for k in range(n)) / n for i in range(self.hb)]
        return c, b, ("img", toks, m, P, Q)

    def _backward_image(self, cache, gc, gb, grads):
        _, toks, m, P, Q = cache
        n = len(toks)
        _add_outer(grads["Wc_img"], gc, m)
        for k, v in enumerate(toks):
            _add_outer(grads["A_img"], [gb[i] * Q[k][i] / n for i in range(self.hb)], v)
            _add_outer(grads["A2_img"], [gb[i] * P[k][i] / n for i in range(self.hb)], v)

    # ---- text ---------------------------------------------------------------
    def encode_text(self, toks):
        L = len(toks)
        m = _mean(toks)
        c = _matvec(self.p["Wc_txt"], m)
        P = [_matvec(self.p["B_txt"], u) for u in toks]
        Q = [_matvec(self.p["B2_txt"], u) for u in toks]
        alpha = self.p["alpha"]
        b = [0.0] * self.hb
        for a, off in zip(alpha, self.offsets):
            for t in range(max(0, -off), min(L, L - off)):
                Pt, Qs = P[t], Q[t + off]
                for i in range(self.hb):
                    b[i] += a * Pt[i] * Qs[i]
        b = [x / L for x in b]
        return c, b, ("txt", toks, m, P, Q)

    def _backward_text(self, cache, gc, gb, grads):
        _, toks, m, P, Q = cache
        L = len(toks)
        _add_outer(grads["Wc_txt"], gc, m)
        alpha = self.p["alpha"]
        gP = [[0.0] * self.hb for _ in range(L)]
        gQ = [[0.0] * self.hb for _ in range(L)]
        for k, (a, off) in enumerate(zip(alpha, self.offsets)):
            ga = 0.0
            for t in range(max(0, -off), min(L, L - off)):
                s = t + off
                Pt, Qs, gPt, gQs = P[t], Q[s], gP[t], gQ[s]
                for i in range(self.hb):
                    g = gb[i] / L
                    ga += g * Pt[i] * Qs[i]
                    gPt[i] += a * g * Qs[i]
                    gQs[i] += a * g * Pt[i]
            grads["alpha"][k] += ga
        for t, u in enumerate(toks):
            _add_outer(grads["B_txt"], gP[t], u)
            _add_outer(grads["B2_txt"], gQ[t], u)

    def backward(self, cache, gc, gb, grads):
        if cache[0] == "img":
            self._backward_image(cache, gc, gb, grads)
        else:
            self._backward_text(cache, gc, gb, grads)


def split_embedding(e, hc):
    return e[:hc], e[hc:]
