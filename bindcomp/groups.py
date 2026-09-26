"""Minimal-change groups: two scenes and two captions that differ by edit ops.

A group is valid only if
  * the 2x2 oracle truth table is exactly the identity (each caption is true
    for its own scene and false for the other), and
  * the metadata-level contract of every operation holds (e.g. a binding swap
    preserves object and attribute multisets but changes the binding).
Member order is randomized so that "member 0" is not always the base scene.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from . import captions as cap
from . import oracle
from .ops import InvalidOp, Op, apply_ops, reachable_in_one
from .scene import Scene


class InvalidGroup(ValueError):
    pass


PROFILE_FIELDS = (
    "shapes_equal",
    "colors_equal",
    "materials_equal",
    "slots_equal",
    "triples_equal",
    "bind_color_equal",
    "bind_material_equal",
    "oid_layout_equal",
    "triple_layout_equal",
)


def diff_profile(s0: Scene, s1: Scene):
    return {
        "shapes_equal": s0.multiset("shape") == s1.multiset("shape"),
        "colors_equal": s0.multiset("color") == s1.multiset("color"),
        "materials_equal": s0.multiset("material") == s1.multiset("material"),
        "slots_equal": s0.multiset("slot") == s1.multiset("slot"),
        "triples_equal": s0.triples() == s1.triples(),
        "bind_color_equal": s0.bindings("color") == s1.bindings("color"),
        "bind_material_equal": s0.bindings("material") == s1.bindings("material"),
        "oid_layout_equal": s0.left_of_oids() == s1.left_of_oids(),
        "triple_layout_equal": s0.left_of_triples() == s1.left_of_triples(),
    }


def expected_profile(op: Op):
    """Required value of each profile field for a single op (None = unconstrained)."""
    e = dict.fromkeys(PROFILE_FIELDS)
    if op.name == "swap_attr":
        attr = op.args[0]
        other = "material" if attr == "color" else "color"
        e.update(shapes_equal=True, colors_equal=True, materials_equal=True, slots_equal=True,
                 triples_equal=False, oid_layout_equal=True)
        e[f"bind_{attr}_equal"] = False
        e[f"bind_{other}_equal"] = True
    elif op.name == "replace_attr":
        attr = op.args[0]
        other = "material" if attr == "color" else "color"
        e.update(shapes_equal=True, slots_equal=True, triples_equal=False, oid_layout_equal=True)
        e[f"{attr}s_equal"] = False
        e[f"{other}s_equal"] = True
        e[f"bind_{other}_equal"] = True
    elif op.name == "replace_shape":
        e.update(shapes_equal=False, colors_equal=True, materials_equal=True, slots_equal=True,
                 triples_equal=False, oid_layout_equal=True)
    elif op.name == "swap_slots":
        e.update(shapes_equal=True, colors_equal=True, materials_equal=True, slots_equal=True,
                 triples_equal=True, bind_color_equal=True, bind_material_equal=True,
                 oid_layout_equal=False, triple_layout_equal=False)
    else:
        raise KeyError(op.name)
    return e


def expected_profile_composed(ops):
    """For a composition only fields preserved by every op are guaranteed preserved."""
    profs = [expected_profile(op) for op in ops]
    e = dict.fromkeys(PROFILE_FIELDS)
    for f in PROFILE_FIELDS:
        if all(p[f] is True for p in profs):
            e[f] = True
    return e


def check_invariants(base: Scene, edited: Scene, ops):
    """List of violated contract fields for ``edited = apply_ops(base, ops)``."""
    prof = diff_profile(base, edited)
    exp = expected_profile(ops[0]) if len(ops) == 1 else expected_profile_composed(ops)
    viol = [f"{f}: expected {exp[f]} got {prof[f]}" for f in PROFILE_FIELDS
            if exp[f] is not None and prof[f] != exp[f]]
    # Counting checks for single content edits: exactly one value swapped out.
    if len(ops) == 1 and ops[0].name in ("replace_attr", "replace_shape"):
        field = ops[0].args[0] if ops[0].name == "replace_attr" else "shape"
        removed = base.multiset(field) - edited.multiset(field)
        added = edited.multiset(field) - base.multiset(field)
        if sum(removed.values()) != 1 or sum(added.values()) != 1:
            viol.append(f"{field} multiset should change by exactly one element")
    if len(ops) > 1 and reachable_in_one(base, edited):
        viol.append("composition is reachable by a single op")
    return viol


def oracle_table(scenes, descs):
    return oracle.truth_matrix(scenes, descs)


def check_oracle(scenes, descs):
    t = oracle_table(scenes, descs)
    viol = []
    if not (t[0][0] and t[1][1]):
        viol.append("caption false for its own scene")
    if t[0][1] or t[1][0]:
        viol.append("caption true for the other scene (group not minimal-contrastive)")
    return viol


def object_list_segment(tokens):
    """Tokens before the relation clause (the list of all object mentions)."""
    tokens = tuple(tokens)
    return tokens[: tokens.index(";")] if ";" in tokens else tokens


def same_word_multiset(captions):
    return Counter(captions[0]) == Counter(captions[1])


def check_caption_minimality(kinds, captions):
    """Word-level contract of a group's two captions.

    Binding/relation-only groups must mention every object with an identical
    word multiset in the object list. The relation clause may still differ in
    word counts (it mentions only two objects), so full-caption equality is
    reported separately by ``same_word_multiset`` rather than enforced; enforcing
    it would force the clause to always mention the edited pair.
    """
    viol = []
    if len(captions[0]) != len(captions[1]):
        viol.append("caption lengths differ")
    seg0, seg1 = (Counter(object_list_segment(c)) for c in captions)
    if set(kinds) <= {"binding", "relation"} and seg0 != seg1:
        viol.append("binding/relation group changed the object-list word multiset")
    if set(kinds) & {"attribute", "object"} and same_word_multiset(captions):
        viol.append("content edit did not change the word multiset")
    return viol


@dataclass
class Group:
    gid: str
    split: str
    scenes: tuple  # (Scene, Scene) in member order
    base_member: int  # which member is the unedited base scene (audit only)
    ops: tuple  # ops mapping base -> edited
    para: cap.Paraphrase
    descs: tuple
    captions: tuple  # token tuples
    render_seeds: tuple

    @property
    def kinds(self):
        return tuple(op.kind for op in self.ops)

    @property
    def kind_label(self):
        return "+".join(sorted(self.kinds))

    @property
    def same_words(self):
        return same_word_multiset(self.captions)

    @property
    def base(self):
        return self.scenes[self.base_member]

    @property
    def edited(self):
        return self.scenes[1 - self.base_member]

    def to_json(self):
        return {
            "gid": self.gid,
            "split": self.split,
            "scenes": [s.to_json() for s in self.scenes],
            "base_member": self.base_member,
            "ops": [op.to_json() for op in self.ops],
            "kinds": list(self.kinds),
            "same_word_multiset": self.same_words,
            "paraphrase": self.para.to_json(),
            "descs": [d.to_json() for d in self.descs],
            "captions": [" ".join(c) for c in self.captions],
            "render_seeds": list(self.render_seeds),
        }

    @staticmethod
    def from_json(d):
        scenes = tuple(Scene.from_json(s) for s in d["scenes"])
        descs = tuple(
            cap.Desc(tuple(tuple(t) for t in x["objects"]), tuple(tuple(r) for r in x["left_of"]))
            for x in d["descs"]
        )
        return Group(
            gid=d["gid"],
            split=d["split"],
            scenes=scenes,
            base_member=d["base_member"],
            ops=tuple(Op.from_json(o) for o in d["ops"]),
            para=cap.Paraphrase.from_json(d["paraphrase"]),
            descs=descs,
            captions=tuple(tuple(c.split(" ")) for c in d["captions"]),
            render_seeds=tuple(d["render_seeds"]),
        )


def validate_group(g: Group):
    viol = []
    base, edited = g.base, g.edited
    try:
        recomputed = apply_ops(base, g.ops)
    except InvalidOp as e:
        return [f"ops not applicable: {e}"]
    if recomputed.canonical_key() != edited.canonical_key():
        viol.append("edited scene != ops(base)")
    viol += check_invariants(base, edited, g.ops)
    viol += check_oracle(g.scenes, g.descs)
    viol += check_caption_minimality(g.kinds, g.captions)
    for s, d, c in zip(g.scenes, g.descs, g.captions):
        if cap.describe(s, g.para) != d:
            viol.append("desc does not match describe(scene, paraphrase)")
        if cap.realize(d, g.para.template, g.para.rel_dir) != c:
            viol.append("caption does not match realize(desc)")
        if cap.parse(c, g.para.template).canonical() != d.canonical():
            viol.append("caption does not parse back to its desc")
    return viol


def make_group(rng: random.Random, base: Scene, ops, para: cap.Paraphrase, gid: str, split: str):
    ops = tuple(ops)
    try:
        edited = apply_ops(base, ops)
    except InvalidOp as e:
        raise InvalidGroup(str(e)) from e
    base_member = rng.randint(0, 1)
    scenes = (base, edited) if base_member == 0 else (edited, base)
    descs = tuple(cap.describe(s, para) for s in scenes)
    captions = tuple(cap.realize(d, para.template, para.rel_dir) for d in descs)
    seeds = (rng.getrandbits(48), rng.getrandbits(48))
    g = Group(gid, split, scenes, base_member, ops, para, descs, captions, seeds)
    viol = validate_group(g)
    if viol:
        raise InvalidGroup("; ".join(viol))
    return g
