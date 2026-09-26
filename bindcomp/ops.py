"""Edit operations on scene graphs.

Four operation kinds, each with a precise metadata-level contract (checked by
``groups.check_invariants``):

* ``binding``   swap_attr(attr, a, b): exchange one attribute between two
                objects. Object and attribute multisets are unchanged; only the
                assignment (binding) differs.
* ``attribute`` replace_attr(attr, a, v): change one attribute value.
* ``object``    replace_shape(a, s): change one object's category.
* ``relation``  swap_slots(a, b): exchange the positions of two objects. All
                multisets and attribute bindings are unchanged; only the
                spatial relation differs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import vocab
from .scene import InvalidScene, Scene

KINDS = ("binding", "attribute", "object", "relation")


class InvalidOp(ValueError):
    pass


@dataclass(frozen=True)
class Op:
    name: str
    args: tuple

    @property
    def kind(self):
        return {
            "swap_attr": "binding",
            "replace_attr": "attribute",
            "replace_shape": "object",
            "swap_slots": "relation",
        }[self.name]

    @property
    def touched_oids(self):
        if self.name == "swap_attr":
            return frozenset(self.args[1:3])
        if self.name == "replace_attr":
            return frozenset([self.args[1]])
        if self.name == "replace_shape":
            return frozenset([self.args[0]])
        if self.name == "swap_slots":
            return frozenset(self.args[0:2])
        raise KeyError(self.name)

    def signature(self):
        return f"{self.name}({','.join(str(a) for a in self.args)})"

    def to_json(self):
        return {"name": self.name, "args": list(self.args)}

    @staticmethod
    def from_json(d):
        return Op(d["name"], tuple(d["args"]))

    def apply(self, scene: Scene) -> Scene:
        objs = {o.oid: o for o in scene.objects}
        if self.name == "swap_attr":
            attr, a, b = self.args
            if attr not in vocab.ATTRS or a == b:
                raise InvalidOp(self.signature())
            va, vb = objs[a].get(attr), objs[b].get(attr)
            if va == vb:
                raise InvalidOp(f"no-op swap: {self.signature()}")
            objs[a] = objs[a].with_(**{attr: vb})
            objs[b] = objs[b].with_(**{attr: va})
        elif self.name == "replace_attr":
            attr, a, value = self.args
            if attr not in vocab.ATTRS or value not in vocab.ATTR_VALUES[attr]:
                raise InvalidOp(self.signature())
            if objs[a].get(attr) == value:
                raise InvalidOp(f"no-op replace: {self.signature()}")
            objs[a] = objs[a].with_(**{attr: value})
        elif self.name == "replace_shape":
            a, value = self.args
            if value not in vocab.SHAPES:
                raise InvalidOp(self.signature())
            if objs[a].shape == value:
                raise InvalidOp(f"no-op replace: {self.signature()}")
            objs[a] = objs[a].with_(shape=value)
        elif self.name == "swap_slots":
            a, b = self.args
            if a == b:
                raise InvalidOp(self.signature())
            sa, sb = objs[a].slot, objs[b].slot
            objs[a] = objs[a].with_(slot=sb)
            objs[b] = objs[b].with_(slot=sa)
        else:
            raise InvalidOp(f"unknown op {self.name}")
        try:
            return Scene(tuple(objs.values()))
        except InvalidScene as e:
            raise InvalidOp(f"{self.signature()} produced invalid scene: {e}") from e


def swap_attr(attr, a, b):
    return Op("swap_attr", (attr, a, b))


def replace_attr(attr, a, value):
    return Op("replace_attr", (attr, a, value))


def replace_shape(a, value):
    return Op("replace_shape", (a, value))


def swap_slots(a, b):
    return Op("swap_slots", (a, b))


def apply_ops(scene: Scene, ops) -> Scene:
    for op in ops:
        scene = op.apply(scene)
    return scene


def enumerate_ops(scene: Scene, kinds=KINDS):
    """All syntactically valid single operations on ``scene`` (may still be no-ops)."""
    oids = scene.oids
    pairs = [(a, b) for i, a in enumerate(oids) for b in oids[i + 1:]]
    out = []
    if "binding" in kinds:
        out += [swap_attr(attr, a, b) for attr in vocab.ATTRS for a, b in pairs]
    if "attribute" in kinds:
        out += [
            replace_attr(attr, a, v)
            for attr in vocab.ATTRS
            for a in oids
            for v in vocab.ATTR_VALUES[attr]
        ]
    if "object" in kinds:
        out += [replace_shape(a, s) for a in oids for s in vocab.SHAPES]
    if "relation" in kinds:
        out += [swap_slots(a, b) for a, b in pairs]
    return out


def applicable_ops(scene: Scene, kinds=KINDS):
    """Operations that change the scene and produce a valid scene."""
    res = []
    for op in enumerate_ops(scene, kinds):
        try:
            op.apply(scene)
        except InvalidOp:
            continue
        res.append(op)
    return res


def sample_op(rng: random.Random, scene: Scene, kind: str):
    ops = applicable_ops(scene, (kind,))
    if not ops:
        raise InvalidOp(f"no applicable {kind} op")
    return rng.choice(ops)


def reachable_in_one(scene: Scene, target: Scene) -> bool:
    """True if some single operation maps ``scene`` to ``target`` (by canonical key)."""
    key = target.canonical_key()
    return any(op.apply(scene).canonical_key() == key for op in applicable_ops(scene))
