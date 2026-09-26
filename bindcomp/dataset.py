"""Leak-controlled split construction and audits.

Leakage units (a key registered to one split may never appear in another):
  * scene           canonical_key of every scene (base and edited). All
                    rendering variants of a scene inherit its split because
                    images are only ever rendered from a group's own scenes.
  * caption content desc.canonical() of every caption. All paraphrases
                    (template / mention order / relation direction) of the same
                    content therefore stay in one split.
Additional split-level constraints:
  * held-out (shape, color) pairs appear only in ``test_heldout_pairs``;
  * the held-out template appears only in ``test_heldout_template``;
  * training/dev/calib use single operations only; ``test_composition`` uses
    two operations that are not reachable by any single operation.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
from math import factorial

from . import captions as cap
from .groups import Group, InvalidGroup, make_group, validate_group
from .ops import KINDS, InvalidOp, sample_op
from .scene import sample_scene

SPLITS = (
    "train",
    "dev",
    "calib",
    "test_iid",
    "test_heldout_pairs",
    "test_composition",
    "test_heldout_template",
)


def heldout_set(cfg):
    return frozenset(tuple(p) for p in cfg["heldout_pairs"])


def kind_schedule(n_ops):
    if n_ops == 1:
        return [(k,) for k in KINDS]
    if n_ops == 2:
        return list(combinations_with_replacement(KINDS, 2))
    raise ValueError("n_ops must be 1 or 2")


def touched_heldout(g: Group, H):
    touched = set()
    for op in g.ops:
        touched |= op.touched_oids
    hits = set()
    for s in g.scenes:
        for o in s.objects:
            if o.oid in touched and (o.shape, o.color) in H:
                hits.add((o.shape, o.color))
    return hits


def any_heldout(g: Group, H):
    return set().union(*(s.shape_color_pairs() for s in g.scenes)) & H


def group_keys(g: Group):
    keys = {("scene", s.canonical_key()) for s in g.scenes}
    keys |= {("desc", d.canonical()) for d in g.descs}
    return keys


def _sample_ops(rng, base, kinds):
    kinds = list(kinds)
    rng.shuffle(kinds)
    ops, s = [], base
    for k in kinds:
        op = sample_op(rng, s, k)
        ops.append(op)
        s = op.apply(s)
    return ops


def build_dataset(cfg, log=None):
    """Return ({split: [Group]}, stats). Deterministic given ``cfg``."""
    H = heldout_set(cfg)
    registry = {}
    out = {}
    stats = {}
    n_obj_choices = [int(k) for k in cfg["n_objects_weights"]]
    n_obj_weights = [cfg["n_objects_weights"][str(k)] for k in n_obj_choices]
    for split in cfg["generation_order"]:
        spec = cfg["splits"][split]
        rng = random.Random(f"{cfg['seed']}:{split}")
        schedule = kind_schedule(spec["n_ops"])
        seen = set()
        groups = []
        rejects = Counter()
        attempts = 0
        while len(groups) < spec["n_groups"]:
            attempts += 1
            if attempts > cfg["max_attempts_per_split"]:
                raise RuntimeError(f"{split}: exceeded max attempts ({dict(rejects)})")
            kinds = schedule[len(groups) % len(schedule)]
            n_obj = rng.choices(n_obj_choices, n_obj_weights)[0]
            forbid = H if spec["heldout"] == "exclude" else frozenset()
            base = sample_scene(rng, n_obj, forbid)
            try:
                ops = _sample_ops(rng, base, kinds)
            except InvalidOp:
                rejects["no_applicable_op"] += 1
                continue
            para = cap.sample_paraphrase(rng, base, spec["templates"], with_relation=True)
            gid = f"{split}-{len(groups):05d}"
            try:
                g = make_group(rng, base, ops, para, gid, split)
            except InvalidGroup as e:
                reason = str(e).split(":")[0][:60]
                rejects[f"invalid_group:{reason}"] += 1
                continue
            if spec["heldout"] == "exclude" and any_heldout(g, H):
                rejects["heldout_pair_present"] += 1
                continue
            if spec["heldout"] == "require_touched" and not touched_heldout(g, H):
                rejects["heldout_pair_not_touched"] += 1
                continue
            keys = group_keys(g)
            if any(registry.get(k, split) != split for k in keys):
                rejects["cross_split_key_conflict"] += 1
                continue
            dedup = (frozenset(s.canonical_key() for s in g.scenes), para.template)
            if dedup in seen:
                rejects["duplicate_group"] += 1
                continue
            seen.add(dedup)
            for k in keys:
                registry[k] = split
            groups.append(g)
        out[split] = groups
        stats[split] = {"n_groups": len(groups), "attempts": attempts, "rejects": dict(sorted(rejects.items()))}
        if log:
            log(f"built {split}: {len(groups)} groups, {attempts} attempts, rejects={dict(rejects)}")
    return {s: out[s] for s in SPLITS if s in out}, stats


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def dumps_jsonl(dataset):
    lines = []
    for split in SPLITS:
        for g in dataset.get(split, []):
            lines.append(json.dumps(g.to_json(), sort_keys=True, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def loads_jsonl(text):
    ds = defaultdict(list)
    for line in text.splitlines():
        if line.strip():
            g = Group.from_json(json.loads(line))
            ds[g.split].append(g)
    return dict(ds)


def dataset_hash(dataset):
    return hashlib.sha256(dumps_jsonl(dataset).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def _rate(flags):
    return round(sum(flags) / len(flags), 4) if flags else None


def audit(dataset, cfg):
    """Structural and leakage audit. Returns dict with ``checks`` (name -> PASS/FAIL detail)."""
    H = heldout_set(cfg)
    checks = {}

    def check(name, ok, detail=None):
        checks[name] = {"status": "PASS" if ok else "FAIL", "detail": detail}

    # 1. every group is valid under its own contract
    bad = {}
    for split, gs in dataset.items():
        for g in gs:
            v = validate_group(g)
            if v:
                bad[g.gid] = v
    check("groups_valid", not bad, {"n_invalid": len(bad), "examples": dict(list(bad.items())[:5])})

    # 2. leakage of scenes / caption contents / caption strings across splits
    owners = defaultdict(set)
    for split, gs in dataset.items():
        for g in gs:
            for k in group_keys(g):
                owners[k].add(split)
            for c in g.captions:
                owners[("caption_string", c)].add(split)
    for kind in ("scene", "desc", "caption_string"):
        leaks = [k for k, v in owners.items() if k[0] == kind and len(v) > 1]
        check(f"no_cross_split_{kind}", not leaks, {"n_leaked": len(leaks)})

    # 3. render seeds unique across the dataset
    seeds = [s for gs in dataset.values() for g in gs for s in g.render_seeds]
    check("render_seeds_unique", len(seeds) == len(set(seeds)), {"n": len(seeds), "n_unique": len(set(seeds))})

    # 4. held-out pairs and templates
    for split, gs in dataset.items():
        spec = cfg["splits"][split]
        n_heldout = sum(1 for g in gs if any_heldout(g, H))
        if spec["heldout"] == "exclude":
            check(f"{split}:no_heldout_pairs", n_heldout == 0, {"n_groups_with_heldout": n_heldout})
        else:
            n_touch = sum(1 for g in gs if touched_heldout(g, H))
            check(f"{split}:heldout_touched_in_all", n_touch == len(gs), {"n_touched": n_touch, "n": len(gs)})
        tmpl = Counter(g.para.template for g in gs)
        check(f"{split}:templates_as_specified", set(tmpl) <= set(spec["templates"]), dict(tmpl))
        nops = Counter(len(g.ops) for g in gs)
        check(f"{split}:n_ops_as_specified", set(nops) == {spec["n_ops"]}, dict(nops))

    # 5. object-set overlap between train and test splits (informational)
    train_sets = {s.object_set_key() for g in dataset.get("train", []) for s in g.scenes}
    overlap = {}
    for split, gs in dataset.items():
        if split == "train" or not gs:
            continue
        sets_ = [s.object_set_key() for g in gs for s in g.scenes]
        overlap[split] = round(sum(k in train_sets for k in sets_) / len(sets_), 4)
    checks["info:object_set_overlap_with_train"] = {"status": "INFO", "detail": overlap}

    # 6. nuisance balance (informational; used to detect sentence-rule shortcuts)
    bal = {}
    for split, gs in dataset.items():
        if not gs:
            continue
        mention_sorted = 0
        expected = 0.0
        for g in gs:
            s = g.scenes[0]
            slots = [s.obj(o).slot for o in g.para.mention_order]
            mention_sorted += slots == sorted(slots)
            expected += 1 / factorial(len(slots))
        bal[split] = {
            "mention_order_is_left_to_right": round(mention_sorted / len(gs), 4),
            "expected_if_independent": round(expected / len(gs), 4),
            "rel_dir_1_rate": round(sum(g.para.rel_dir for g in gs) / len(gs), 4),
            "base_member_1_rate": round(sum(g.base_member for g in gs) / len(gs), 4),
            "kinds": dict(Counter(g.kind_label for g in gs)),
            "n_objects": dict(Counter(len(g.scenes[0]) for g in gs)),
            "same_word_multiset_rate_binding_relation": _rate(
                [g.same_words for g in gs if set(g.kinds) <= {"binding", "relation"}]),
        }
    checks["info:balance"] = {"status": "INFO", "detail": bal}

    n_fail = sum(1 for c in checks.values() if c["status"] == "FAIL")
    return {"checks": checks, "n_fail": n_fail}
