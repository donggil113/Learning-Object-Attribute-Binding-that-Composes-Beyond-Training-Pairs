"""Encoded view of a group: proxy token features for both images and captions.

Also builds a meaning-preserving paraphrase ("hard positive", in the sense of
Kamath et al., ECCV 2024) of each caption: reversed mention order and converse
relation direction, same template, same description graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import captions as cap
from .groups import Group
from .oracle import satisfies


@dataclass
class EncGroup:
    group: Group
    enc: object  # ProxyEncoder
    img: tuple  # token lists for member 0 / 1 (own render seeds)
    txt: tuple
    txt_hp: tuple  # hard-positive paraphrases
    captions_hp: tuple

    @property
    def gid(self):
        return self.group.gid

    @property
    def kinds(self):
        return self.group.kinds

    @property
    def kind_label(self):
        return self.group.kind_label

    @property
    def scenes(self):
        return self.group.scenes

    @property
    def descs(self):
        return self.group.descs

    @property
    def same_words(self):
        return self.group.same_words


def hard_positive_paraphrase(para: cap.Paraphrase) -> cap.Paraphrase:
    return cap.Paraphrase(para.template, tuple(reversed(para.mention_order)), para.rel_pair, 1 - para.rel_dir)


def encode_group(g: Group, enc) -> EncGroup:
    img = tuple(enc.image_tokens(s, seed) for s, seed in zip(g.scenes, g.render_seeds))
    txt = tuple(enc.text_tokens(c) for c in g.captions)
    hp = hard_positive_paraphrase(g.para)
    caps_hp = []
    for s, d in zip(g.scenes, g.descs):
        d_hp = cap.describe(s, hp)
        assert d_hp.canonical() == d.canonical()
        caps_hp.append(cap.realize(d_hp, hp.template, hp.rel_dir))
    for m in (0, 1):
        d_hp = cap.parse(caps_hp[m], hp.template)
        assert satisfies(g.scenes[m], d_hp) and not satisfies(g.scenes[1 - m], d_hp)
    txt_hp = tuple(enc.text_tokens(c) for c in caps_hp)
    return EncGroup(g, enc, img, txt, txt_hp, tuple(caps_hp))


def encode_groups(groups, enc):
    return [encode_group(g, enc) for g in groups]


def image_tokens_for(eg: EncGroup, m: int, same_seed=False):
    """Image tokens of member m; with ``same_seed`` both members share member 0's render seed."""
    if not same_seed:
        return eg.img[m]
    return eg.enc.image_tokens(eg.scenes[m], eg.group.render_seeds[0])
