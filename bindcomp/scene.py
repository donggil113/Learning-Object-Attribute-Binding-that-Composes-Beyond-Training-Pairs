"""Scene graphs: a small set of objects with attributes and discrete positions.

An object is identified by an ``oid`` that is stable under edits; ``oid`` is
bookkeeping only and is excluded from every semantic key.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, replace

from . import vocab


class InvalidScene(ValueError):
    pass


@dataclass(frozen=True)
class Obj:
    oid: int
    shape: str
    color: str
    material: str
    slot: int

    @property
    def triple(self):
        return (self.shape, self.color, self.material)

    def get(self, field):
        return getattr(self, field)

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class Scene:
    objects: tuple  # tuple[Obj, ...], sorted by oid

    def __post_init__(self):
        objs = tuple(sorted(self.objects, key=lambda o: o.oid))
        object.__setattr__(self, "objects", objs)
        oids = [o.oid for o in objs]
        slots = [o.slot for o in objs]
        triples = [o.triple for o in objs]
        if len(set(oids)) != len(oids):
            raise InvalidScene("duplicate oid")
        if len(set(slots)) != len(slots):
            raise InvalidScene("two objects share a slot")
        if len(set(triples)) != len(triples):
            # Identical objects would make definite descriptions ambiguous.
            raise InvalidScene("two objects have identical (shape, color, material)")
        for o in objs:
            if o.shape not in vocab.SHAPES:
                raise InvalidScene(f"unknown shape {o.shape}")
            if o.color not in vocab.COLORS:
                raise InvalidScene(f"unknown color {o.color}")
            if o.material not in vocab.MATERIALS:
                raise InvalidScene(f"unknown material {o.material}")
            if not 0 <= o.slot < vocab.N_SLOTS:
                raise InvalidScene(f"slot out of range {o.slot}")

    # ---- access -----------------------------------------------------------
    def __len__(self):
        return len(self.objects)

    def obj(self, oid):
        for o in self.objects:
            if o.oid == oid:
                return o
        raise KeyError(oid)

    @property
    def oids(self):
        return tuple(o.oid for o in self.objects)

    # ---- semantic keys ----------------------------------------------------
    def canonical_key(self):
        """Full semantic identity: objects with attributes and positions."""
        return tuple(sorted((o.slot, o.shape, o.color, o.material) for o in self.objects))

    def object_set_key(self):
        """Position-free identity: which objects exist (attributes bound)."""
        return tuple(sorted(o.triple for o in self.objects))

    def content_key(self):
        """Edit-orbit identity: object/attribute multisets without their assignment.

        Invariant under binding swaps and relation swaps, so every scene reachable
        from this one by binding-necessary edits shares the key.
        """
        return tuple(tuple(sorted(o.get(f) for o in self.objects)) for f in ("shape", "color", "material"))

    def multiset(self, field):
        return Counter(o.get(field) for o in self.objects)

    def triples(self):
        return Counter(o.triple for o in self.objects)

    def bindings(self, attr):
        """Multiset of (shape, attribute value) assignments."""
        return Counter((o.shape, o.get(attr)) for o in self.objects)

    def shape_color_pairs(self):
        return {(o.shape, o.color) for o in self.objects}

    def left_of_triples(self):
        """Set of (triple_x, triple_y) with x strictly left of y."""
        return {
            (a.triple, b.triple)
            for a in self.objects
            for b in self.objects
            if a.slot < b.slot
        }

    def left_of_oids(self):
        return {(a.oid, b.oid) for a in self.objects for b in self.objects if a.slot < b.slot}

    def to_json(self):
        return [
            {"oid": o.oid, "shape": o.shape, "color": o.color, "material": o.material, "slot": o.slot}
            for o in self.objects
        ]

    @staticmethod
    def from_json(items):
        return Scene(tuple(Obj(**d) for d in items))


def sample_scene(rng: random.Random, n_objects: int, forbidden_pairs=frozenset(), max_tries=1000):
    """Uniformly sample a valid scene avoiding ``forbidden_pairs`` of (shape, color)."""
    if not 1 <= n_objects <= vocab.N_SLOTS:
        raise ValueError("n_objects out of range")
    allowed = [
        (s, c, m)
        for s in vocab.SHAPES
        for c in vocab.COLORS
        for m in vocab.MATERIALS
        if (s, c) not in forbidden_pairs
    ]
    for _ in range(max_tries):
        triples = rng.sample(allowed, n_objects)
        slots = rng.sample(range(vocab.N_SLOTS), n_objects)
        objs = tuple(
            Obj(oid=i, shape=t[0], color=t[1], material=t[2], slot=slots[i])
            for i, t in enumerate(triples)
        )
        try:
            return Scene(objs)
        except InvalidScene:
            continue
    raise RuntimeError("could not sample a valid scene")
