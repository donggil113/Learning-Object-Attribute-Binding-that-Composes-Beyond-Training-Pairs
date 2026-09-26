"""Ground-truth semantics: does a scene satisfy a description?

Existential semantics: there must be an injective assignment of described
objects to scene objects such that every specified attribute matches and every
``left_of`` relation holds (strictly smaller slot).
"""

from __future__ import annotations

from itertools import permutations

from .captions import Desc
from .scene import Scene


def _obj_matches(obj, triple):
    shape, color, material = triple
    return (
        (shape is None or obj.shape == shape)
        and (color is None or obj.color == color)
        and (material is None or obj.material == material)
    )


def satisfies(scene: Scene, desc: Desc) -> bool:
    k = len(desc.objects)
    if k > len(scene.objects):
        return False
    for assign in permutations(scene.objects, k):
        if not all(_obj_matches(o, t) for o, t in zip(assign, desc.objects)):
            continue
        if all(assign[i].slot < assign[j].slot for i, j in desc.left_of):
            return True
    return False


def truth_matrix(scenes, descs):
    return [[satisfies(s, d) for d in descs] for s in scenes]
