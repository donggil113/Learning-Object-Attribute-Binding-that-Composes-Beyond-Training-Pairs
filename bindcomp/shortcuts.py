"""Scorers used for evaluation, including single-modality (blind) shortcuts.

HeadScorer(channels=all|content|binding)  trained head; ``content`` is the
    bag-level channel alone (what a binding-blind model can do per op kind).
TextOnlyScorer / ImageOnlyScorer  logistic regression on one modality only,
    trained on the same train pairs (positives, group-partner negatives and
    random in-batch negatives). On 2x2 groups their group metrics are 0 by
    construction (all comparisons tie); their pair AUC is the informative number.
RandomScorer / OracleScorer  metric sanity bounds.

``blind_detectability`` trains a single-modality classifier to tell the base
scene/caption from the edited one. AUC near 0.5 means the generator leaves no
renderer or sentence-rule trace of which side was edited.
"""

from __future__ import annotations

import random
from collections import Counter
from math import exp

from .evaluate import auc
from .oracle import satisfies


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class HeadScorer:
    def __init__(self, head, channels="all"):
        self.head, self.channels = head, channels
        self.name = f"head[{channels}]"

    def _emb(self, c, b):
        if self.channels == "content":
            return c
        if self.channels == "binding":
            return b
        return c + b

    def score_group(self, eg):
        eI = [self._emb(*self.head.encode_image(eg.img[m])[:2]) for m in (0, 1)]
        eT = [self._emb(*self.head.encode_text(eg.txt[m])[:2]) for m in (0, 1)]
        eH = [self._emb(*self.head.encode_text(eg.txt_hp[m])[:2]) for m in (0, 1)]
        S = [[_dot(eI[i], eT[t]) for t in (0, 1)] for i in (0, 1)]
        return S, [_dot(eI[i], eH[i]) for i in (0, 1)]


class RandomScorer:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(f"random-scorer:{seed}")

    def score_group(self, eg):
        S = [[self.rng.random() for _ in (0, 1)] for _ in (0, 1)]
        return S, [self.rng.random() for _ in (0, 1)]


class OracleScorer:
    name = "oracle"

    def score_group(self, eg):
        S = [[float(satisfies(eg.scenes[i], eg.descs[t])) for t in (0, 1)] for i in (0, 1)]
        return S, [1.0, 1.0]


# ---------------------------------------------------------------------------
# Logistic regression on sparse features (stdlib)
# ---------------------------------------------------------------------------

def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + exp(-z))
    e = exp(z)
    return e / (1.0 + e)


class LogReg:
    def __init__(self, l2=1e-3, lr=0.1, epochs=20, seed=0):
        self.l2, self.lr, self.epochs, self.seed = l2, lr, epochs, seed
        self.w = {}
        self.b = 0.0

    def fit(self, X, y):
        rng = random.Random(f"logreg:{self.seed}")
        idx = list(range(len(X)))
        for ep in range(self.epochs):
            rng.shuffle(idx)
            lr = self.lr / (1 + ep)
            for i in idx:
                p = self.predict(X[i])
                g = p - y[i]
                for k, v in X[i].items():
                    w = self.w.get(k, 0.0)
                    self.w[k] = w - lr * (g * v + self.l2 * w)
                self.b -= lr * g
        return self

    def decision(self, x):
        return self.b + sum(self.w.get(k, 0.0) * v for k, v in x.items())

    def predict(self, x):
        return _sigmoid(self.decision(x))


def text_features(tokens):
    f = Counter(f"u:{w}" for w in tokens)
    f.update(f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:]))
    n = len(tokens)
    return {k: v / n for k, v in f.items()}


def image_features(toks):
    n, d = len(toks), len(toks[0])
    f = {}
    for i in range(d):
        col = [t[i] for t in toks]
        f[f"m{i}"] = sum(col) / n
        f[f"q{i}"] = sum(x * x for x in col) / n
    return f


def _train_pairs(enc_groups, n_random, seed):
    """(image member, caption member) training pairs with oracle labels."""
    rng = random.Random(f"blind-pairs:{seed}")
    out = []
    for eg in enc_groups:
        for i in (0, 1):
            for t in (0, 1):
                out.append((eg, i, eg, t, int(satisfies(eg.scenes[i], eg.descs[t]))))
            for _ in range(n_random):
                other = rng.choice(enc_groups)
                t = rng.randint(0, 1)
                out.append((eg, i, other, t, int(satisfies(eg.scenes[i], other.descs[t]))))
    return out


class TextOnlyScorer:
    name = "text_only"

    def fit(self, enc_groups, n_random=2, seed=0):
        pairs = _train_pairs(enc_groups, n_random, seed)
        X = [text_features(eg_t.group.captions[t]) for _, _, eg_t, t, _ in pairs]
        y = [lab for *_, lab in pairs]
        self.model = LogReg(seed=seed).fit(X, y)
        return self

    def score_group(self, eg):
        f = [self.model.decision(text_features(eg.group.captions[t])) for t in (0, 1)]
        h = [self.model.decision(text_features(eg.captions_hp[t])) for t in (0, 1)]
        return [[f[0], f[1]], [f[0], f[1]]], h


class ImageOnlyScorer:
    name = "image_only"

    def fit(self, enc_groups, n_random=2, seed=0):
        pairs = _train_pairs(enc_groups, n_random, seed)
        X = [image_features(eg_i.img[i]) for eg_i, i, _, _, _ in pairs]
        y = [lab for *_, lab in pairs]
        self.model = LogReg(seed=seed).fit(X, y)
        return self

    def score_group(self, eg):
        f = [self.model.decision(image_features(eg.img[i])) for i in (0, 1)]
        return [[f[0], f[0]], [f[1], f[1]]], [f[0], f[1]]


def blind_detectability(train_groups, eval_groups, modality, seed=0):
    """AUC of a single-modality classifier predicting 'is this the base (unedited) side?'."""
    def feats(eg, m):
        if modality == "text":
            return text_features(eg.group.captions[m])
        return image_features(eg.img[m])

    def data(groups):
        X, y = [], []
        for eg in groups:
            for m in (0, 1):
                X.append(feats(eg, m))
                y.append(int(m == eg.group.base_member))
        return X, y

    Xtr, ytr = data(train_groups)
    Xev, yev = data(eval_groups)
    model = LogReg(seed=seed).fit(Xtr, ytr)
    scores = [model.decision(x) for x in Xev]
    return {"modality": modality, "auc": auc(scores, yev), "n_train": len(ytr), "n_eval": len(yev),
            "scores": scores, "labels": yev}
