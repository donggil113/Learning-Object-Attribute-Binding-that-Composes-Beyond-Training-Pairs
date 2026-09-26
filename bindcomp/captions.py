"""Description graphs, paraphrase templates, realization and parsing.

A ``Desc`` is the semantic content of a caption: a list of (possibly partial)
object descriptions plus ``left_of`` relations between them. Truth of a
caption for a scene is decided by ``oracle.satisfies(scene, desc)``, never by
string matching.

Templates (all words are in ``vocab.WORDS``):
  0 prenominal : "a red metal cube"
  1 relative   : "a cube that is red and metal"
  2 mixed      : "a metal cube that is red"        (held-out template)
Caption = NP_1 , NP_2 [, NP_3] [; NP_def(x) is left of NP_def(y)]
with the relation optionally realized in the converse form "y is right of x".
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import vocab
from .scene import Scene

TEMPLATES = (0, 1, 2)
NP_LEN = {0: 4, 1: 7, 2: 6}


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Desc:
    # (shape, color, material); None means "unspecified" (only for hand-built
    # partial descriptions; generated captions are always fully specified).
    objects: tuple
    left_of: tuple = ()  # tuple[(i, j)] indices into objects

    def canonical(self):
        order = sorted(range(len(self.objects)), key=lambda i: tuple("" if v is None else v for v in self.objects[i]))
        pos = {old: new for new, old in enumerate(order)}
        objs = tuple(self.objects[i] for i in order)
        rels = tuple(sorted((pos[i], pos[j]) for i, j in self.left_of))
        return (objs, rels)

    def to_json(self):
        return {"objects": [list(t) for t in self.objects], "left_of": [list(r) for r in self.left_of]}


@dataclass(frozen=True)
class Paraphrase:
    template: int
    mention_order: tuple  # oids in mention order
    rel_pair: tuple | None  # unordered pair of oids the relation clause is about
    rel_dir: int  # 0: "x is left of y"; 1: "y is right of x"

    def to_json(self):
        return {
            "template": self.template,
            "mention_order": list(self.mention_order),
            "rel_pair": None if self.rel_pair is None else list(self.rel_pair),
            "rel_dir": self.rel_dir,
        }

    @staticmethod
    def from_json(d):
        return Paraphrase(
            d["template"],
            tuple(d["mention_order"]),
            None if d["rel_pair"] is None else tuple(d["rel_pair"]),
            d["rel_dir"],
        )


def sample_paraphrase(rng: random.Random, scene: Scene, templates, with_relation=True):
    order = list(scene.oids)
    rng.shuffle(order)
    rel_pair = None
    if with_relation and len(order) >= 2:
        rel_pair = tuple(sorted(rng.sample(list(scene.oids), 2)))
    return Paraphrase(rng.choice(tuple(templates)), tuple(order), rel_pair, rng.randint(0, 1))


def describe(scene: Scene, para: Paraphrase) -> Desc:
    """Full description of ``scene`` under paraphrase choices ``para``.

    ``mention_order`` and ``rel_pair`` refer to object identities (oids), so the
    same paraphrase applied to an edited scene yields a minimally different
    description (same order, same pair, updated attributes / direction).
    """
    if sorted(para.mention_order) != sorted(scene.oids):
        raise ValueError("mention order does not match scene objects")
    idx = {oid: i for i, oid in enumerate(para.mention_order)}
    objects = tuple(scene.obj(oid).triple for oid in para.mention_order)
    rels = ()
    if para.rel_pair is not None:
        a, b = para.rel_pair
        if scene.obj(a).slot < scene.obj(b).slot:
            rels = ((idx[a], idx[b]),)
        else:
            rels = ((idx[b], idx[a]),)
    return Desc(objects, rels)


def _np(triple, template, det):
    shape, color, material = triple
    if None in triple:
        raise ValueError("cannot realize a partial description")
    if template == 0:
        return [det, color, material, shape]
    if template == 1:
        return [det, shape, "that", "is", color, "and", material]
    if template == 2:
        return [det, material, shape, "that", "is", color]
    raise ValueError(f"unknown template {template}")


def realize(desc: Desc, template: int, rel_dir: int = 0):
    toks = []
    for k, t in enumerate(desc.objects):
        if k:
            toks.append(",")
        toks += _np(t, template, "a")
    if len(desc.left_of) > 1:
        raise ValueError("at most one relation clause is supported")
    for i, j in desc.left_of:
        toks.append(";")
        if rel_dir == 0:
            toks += _np(desc.objects[i], template, "the") + ["is", "left", "of"] + _np(desc.objects[j], template, "the")
        else:
            toks += _np(desc.objects[j], template, "the") + ["is", "right", "of"] + _np(desc.objects[i], template, "the")
    for w in toks:
        assert w in vocab.WORDS, w
    return tuple(toks)


def _parse_np(toks, template, det):
    n = NP_LEN[template]
    if len(toks) != n or toks[0] != det:
        raise ParseError(f"bad NP {toks}")
    if template == 0:
        _, color, material, shape = toks
    elif template == 1:
        _, shape, w1, w2, color, w3, material = toks
        if (w1, w2, w3) != ("that", "is", "and"):
            raise ParseError(f"bad NP {toks}")
    else:
        _, material, shape, w1, w2, color = toks
        if (w1, w2) != ("that", "is"):
            raise ParseError(f"bad NP {toks}")
    if shape not in vocab.SHAPES or color not in vocab.COLORS or material not in vocab.MATERIALS:
        raise ParseError(f"bad NP {toks}")
    return (shape, color, material)


def parse(tokens, template: int) -> Desc:
    """Inverse of ``realize`` for a known template (used to prove captions encode bindings)."""
    toks = list(tokens)
    if toks.count(";") > 1:
        raise ParseError("more than one relation clause")
    rel_toks = None
    if ";" in toks:
        k = toks.index(";")
        toks, rel_toks = toks[:k], toks[k + 1:]
    n = NP_LEN[template]
    objects = []
    pos = 0
    while pos < len(toks):
        if objects:
            if toks[pos] != ",":
                raise ParseError("expected ','")
            pos += 1
        objects.append(_parse_np(toks[pos:pos + n], template, "a"))
        pos += n
    rels = ()
    if rel_toks is not None:
        if len(rel_toks) != 2 * n + 3:
            raise ParseError("bad relation clause length")
        first = _parse_np(rel_toks[:n], template, "the")
        is_, direction, of = rel_toks[n:n + 3]
        second = _parse_np(rel_toks[n + 3:], template, "the")
        if is_ != "is" or of != "of" or direction not in ("left", "right"):
            raise ParseError("bad relation clause")
        if objects.count(first) != 1 or objects.count(second) != 1:
            raise ParseError("relation refers to an ambiguous or missing object")
        i, j = objects.index(first), objects.index(second)
        rels = ((i, j),) if direction == "left" else ((j, i),)
    return Desc(tuple(objects), rels)
