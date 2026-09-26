"""Frozen *proxy* features standing in for frozen image/text encoders.

This is NOT an image renderer and NOT a pretrained encoder. It produces
token-level feature vectors from scene metadata so that the head, losses and
evaluation can be exercised on CPU without numpy/torch or model weights.

Image side: one token per object, the sum of fixed random embeddings of its
shape, color, material and slot, plus per-image nuisance (a global "lighting"
offset and per-object jitter) drawn from the image's render seed; object order
is shuffled per render. Binding is therefore present *within* each token, and
lost by any linear pooling -- which is the property the binding head must
exploit. This is an oracle-object-token assumption: real encoders do not hand
over one clean token per object (see STATUS.md, "unverified").

Text side: fixed random word embeddings plus a scaled position embedding.
Base and edited scenes go through exactly the same function with independent
seeds (no renderer trace distinguishes them by construction; see the blind
detectability audit).
"""

from __future__ import annotations

import random
from math import sqrt

from . import vocab


def _gvec(rng, d, std):
    return [rng.gauss(0.0, std) for _ in range(d)]


class ProxyEncoder:
    def __init__(self, dim=24, seed=0, jitter=0.1, lighting=0.1, pos_scale=0.5, max_len=64):
        self.dim = dim
        self.jitter = jitter
        self.lighting = lighting
        rng = random.Random(f"proxy-encoder:{seed}")
        std = 1.0 / sqrt(dim)
        self.shape = {s: _gvec(rng, dim, std) for s in vocab.SHAPES}
        self.color = {c: _gvec(rng, dim, std) for c in vocab.COLORS}
        self.material = {m: _gvec(rng, dim, std) for m in vocab.MATERIALS}
        self.slot = [_gvec(rng, dim, std) for _ in range(vocab.N_SLOTS)]
        self.word = {w: _gvec(rng, dim, std) for w in vocab.WORDS}
        self.pos = [[pos_scale * x for x in _gvec(rng, dim, std)] for _ in range(max_len)]
        self.config = dict(dim=dim, seed=seed, jitter=jitter, lighting=lighting, pos_scale=pos_scale,
                           max_len=max_len)

    def image_tokens(self, scene, render_seed):
        r = random.Random(render_seed)
        light = _gvec(r, self.dim, self.lighting)
        objs = list(scene.objects)
        r.shuffle(objs)
        toks = []
        for o in objs:
            parts = (self.shape[o.shape], self.color[o.color], self.material[o.material], self.slot[o.slot])
            toks.append([
                parts[0][i] + parts[1][i] + parts[2][i] + parts[3][i] + light[i] + r.gauss(0.0, self.jitter)
                for i in range(self.dim)
            ])
        return toks

    def text_tokens(self, tokens):
        if len(tokens) > len(self.pos):
            raise ValueError("caption longer than max_len")
        return [[a + b for a, b in zip(self.word[w], self.pos[t])] for t, w in enumerate(tokens)]
