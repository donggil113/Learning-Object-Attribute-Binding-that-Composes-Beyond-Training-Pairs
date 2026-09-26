"""Pixel-path feature groups, scorers and controls.

Model-facing fields of ``FeatGroup`` (img*, txt*, captions*) are computed only
from rendered pixels and caption strings. ``meta`` (the Group: scenes, ops,
descriptions, IDs) is kept for labels, false-negative masks, subset reporting
and audits, and is never read by a model scorer; tests poison it to check this.

Provenance of every tensor used in this stage:
  img, img_pooled         render(scene) -> PIL RGB -> frozen CLIP  (pixels only)
  txt, txt_hp, *_pooled   caption string -> frozen CLIP            (text only)
  labels / masks          oracle.satisfies(meta.scenes, meta.descs) (supervision, not input)
  proxy tokens            render.ProxyEncoder(scene)  -> ORACLE CONTROL only
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import torch

from . import captions as cap
from .encode import hard_positive_paraphrase
from .oracle import satisfies
from .pixel_render import render
from .shortcuts import LogReg


@dataclass
class FeatGroup:
    gid: str
    kind_label: str
    same_words: bool
    img: torch.Tensor  # (2, 50, 512) standardized tokens
    img_pooled: torch.Tensor  # (2, 512) raw CLIP embedding
    txt: list  # 2 x (L, 512) standardized tokens
    txt_hp: list
    txt_pooled: torch.Tensor  # (2, 512)
    txt_hp_pooled: torch.Tensor
    captions: tuple  # token tuples of the raw caption strings
    captions_hp: tuple
    meta: object  # Group: labels / masks / audit only

    @property
    def scenes(self):
        return self.meta.scenes

    @property
    def descs(self):
        return self.meta.descs


def hp_captions(g):
    hp = hard_positive_paraphrase(g.para)
    return tuple(cap.realize(cap.describe(s, hp), hp.template, hp.rel_dir) for s in g.scenes)


def render_members(groups, seed_offset=None):
    """Render both members of each group. ``seed_offset`` gives an independent nuisance variant."""
    imgs = []
    for g in groups:
        for m in (0, 1):
            seed = g.render_seeds[m] if seed_offset is None else f"{g.render_seeds[m]}:{seed_offset}"
            imgs.append(render(g.scenes[m], seed))
    return imgs


def build_feature_groups(groups, encoder, images=None, log=None, text_batch=None):
    """Render (unless given) + encode. The encoder only ever sees PIL images and caption strings."""
    images = render_members(groups) if images is None else images
    img_pooled, img_tok = encoder.encode_images(images)
    caps_hp = [hp_captions(g) for g in groups]
    texts = [" ".join(c) for g in groups for c in g.captions]
    texts_hp = [" ".join(c) for ch in caps_hp for c in ch]
    t_pooled, t_tok = encoder.encode_texts(texts, batch=text_batch) if text_batch else encoder.encode_texts(texts)
    h_pooled, h_tok = encoder.encode_texts(texts_hp, batch=text_batch) if text_batch else encoder.encode_texts(texts_hp)
    out = []
    for i, g in enumerate(groups):
        a, b = 2 * i, 2 * i + 2
        out.append(FeatGroup(g.gid, g.kind_label, g.same_words, img_tok[a:b].clone(), img_pooled[a:b].clone(),
                             [t_tok[a], t_tok[a + 1]], [h_tok[a], h_tok[a + 1]], t_pooled[a:b].clone(),
                             h_pooled[a:b].clone(), tuple(g.captions), caps_hp[i], g))
    if log:
        log(f"encoded {len(images)} images and {len(texts) + len(texts_hp)} captions")
    return out


def standardizer(train_fgs):
    """Per-dimension mean/std of image and text tokens from the TRAIN panel only."""
    it = torch.cat([fg.img.reshape(-1, fg.img.shape[-1]) for fg in train_fgs])
    tt = torch.cat([t for fg in train_fgs for t in fg.txt])
    stats = {"img": (it.mean(0), it.std(0) + 1e-6), "txt": (tt.mean(0), tt.std(0) + 1e-6)}
    return stats


def apply_standardizer(fgs, stats):
    (mi, si), (mt, st) = stats["img"], stats["txt"]
    return [replace(fg, img=(fg.img - mi) / si, txt=[(t - mt) / st for t in fg.txt],
                    txt_hp=[(t - mt) / st for t in fg.txt_hp]) for fg in fgs]


# ---------------------------------------------------------------------------
# Scorers and controls
# ---------------------------------------------------------------------------

class ZeroShotClipScorer:
    """Frozen CLIP cosine of pooled embeddings; no training (encoder reference)."""

    name = "zero_shot_clip"
    provenance = "pixels + caption text"

    def score_group(self, fg):
        i = torch.nn.functional.normalize(fg.img_pooled, dim=-1)
        t = torch.nn.functional.normalize(fg.txt_pooled, dim=-1)
        h = torch.nn.functional.normalize(fg.txt_hp_pooled, dim=-1)
        return (i @ t.T).tolist(), [float(i[k] @ h[k]) for k in (0, 1)]


def _pooled_feats(v):
    v = torch.nn.functional.normalize(v, dim=-1)
    return {f"d{k}": float(x) for k, x in enumerate(v)}


class ClipImageOnlyScorer:
    """Blind control: logistic regression on the pooled image embedding only."""

    name = "image_only_clip"
    provenance = "pixels only"

    def fit(self, fgs, seed=0):
        X, y = [], []
        for fg in fgs:
            for i in (0, 1):
                for t in (0, 1):
                    X.append(_pooled_feats(fg.img_pooled[i]))
                    y.append(int(satisfies(fg.scenes[i], fg.descs[t])))  # label (supervision)
        self.model = LogReg(seed=seed).fit(X, y)
        return self

    def score_group(self, fg):
        f = [self.model.decision(_pooled_feats(fg.img_pooled[i])) for i in (0, 1)]
        return [[f[0], f[0]], [f[1], f[1]]], [f[0], f[1]]


def shuffled(fgs, what, seed=0):
    """Control: replace images or captions of each group by those of another group (derangement)."""
    rng = random.Random(f"shuffle-{what}:{seed}")
    n = len(fgs)
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(p != i for i, p in enumerate(perm)):
            break
    out = []
    for i, fg in enumerate(fgs):
        o = fgs[perm[i]]
        if what == "image":
            out.append(replace(fg, img=o.img, img_pooled=o.img_pooled))
        elif what == "text":
            out.append(replace(fg, txt=o.txt, txt_hp=o.txt_hp, txt_pooled=o.txt_pooled,
                               txt_hp_pooled=o.txt_hp_pooled, captions=o.captions, captions_hp=o.captions_hp))
        else:
            raise ValueError(what)
    return out


def base_vs_edited_detectability(train_fgs, eval_fgs, seed=0):
    """AUC of a pooled-image logistic regression predicting the unedited side (renderer trace)."""
    from .evaluate import auc

    def data(fgs):
        X, y = [], []
        for fg in fgs:
            for m in (0, 1):
                X.append(_pooled_feats(fg.img_pooled[m]))
                y.append(int(m == fg.meta.base_member))
        return X, y

    Xtr, ytr = data(train_fgs)
    Xev, yev = data(eval_fgs)
    model = LogReg(seed=seed).fit(Xtr, ytr)
    scores = [model.decision(x) for x in Xev]
    return auc(scores, yev), scores, yev


def cosine_distance(a, b):
    return 1.0 - float(torch.nn.functional.cosine_similarity(a, b, dim=-1))
