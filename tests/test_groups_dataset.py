import copy
import json
import random
import unittest
from pathlib import Path

from bindcomp import captions as cap
from bindcomp.dataset import (SPLITS, audit, build_dataset, dataset_hash, dumps_jsonl, group_keys,
                              loads_jsonl)
from bindcomp.groups import InvalidGroup, make_group, validate_group
from bindcomp.ops import replace_shape, swap_attr, swap_slots
from bindcomp.scene import Obj, Scene

ROOT = Path(__file__).resolve().parents[1]


def small_cfg():
    cfg = json.loads((ROOT / "configs" / "data_v0.json").read_text())
    for split, spec in cfg["splits"].items():
        spec["n_groups"] = 40 if split == "train" else 16
    return cfg


def scene3():
    return Scene((
        Obj(0, "cube", "red", "metal", 0),
        Obj(1, "sphere", "blue", "rubber", 1),
        Obj(2, "cone", "green", "metal", 2),
    ))


class TestGroups(unittest.TestCase):
    def test_valid_binding_group(self):
        rng = random.Random(0)
        s = scene3()
        para = cap.Paraphrase(0, (0, 1, 2), (0, 1), 0)
        g = make_group(rng, s, (swap_attr("color", 0, 1),), para, "g", "train")
        self.assertEqual(validate_group(g), [])
        self.assertTrue(g.same_words)  # clause mentions both swapped objects

    def test_relation_group_needs_clause_on_moved_pair(self):
        rng = random.Random(0)
        s = scene3()
        # Swapping cube and sphere does not change "cube left of cone": captions
        # would be identical -> both off-diagonal cells true -> rejected.
        para = cap.Paraphrase(0, (0, 1, 2), (0, 2), 0)
        with self.assertRaises(InvalidGroup):
            make_group(rng, s, (swap_slots(0, 1),), para, "g", "train")
        para_ok = cap.Paraphrase(0, (0, 1, 2), (0, 1), 0)
        g = make_group(rng, s, (swap_slots(0, 1),), para_ok, "g", "train")
        self.assertEqual(validate_group(g), [])

    def test_corrupted_group_is_detected(self):
        rng = random.Random(0)
        s = scene3()
        para = cap.Paraphrase(0, (0, 1, 2), (0, 1), 0)
        g = make_group(rng, s, (replace_shape(2, "cube"),), para, "g", "train")
        bad = copy.copy(g)
        bad.captions = (g.captions[0], g.captions[0])
        self.assertTrue(validate_group(bad))
        bad2 = copy.copy(g)
        bad2.ops = (swap_attr("color", 0, 1),)
        self.assertTrue(validate_group(bad2))

    def test_member_order_is_randomized(self):
        rng = random.Random(5)
        s = scene3()
        para = cap.Paraphrase(0, (0, 1, 2), (0, 1), 0)
        members = [make_group(rng, s, (swap_attr("color", 0, 1),), para, "g", "t").base_member
                   for _ in range(200)]
        self.assertTrue(60 < sum(members) < 140)


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = small_cfg()
        cls.ds, cls.stats = build_dataset(cls.cfg)

    def test_audit_passes(self):
        rep = audit(self.ds, self.cfg)
        failed = {k: v for k, v in rep["checks"].items() if v["status"] == "FAIL"}
        self.assertEqual(failed, {})

    def test_deterministic(self):
        ds2, _ = build_dataset(self.cfg)
        self.assertEqual(dataset_hash(self.ds), dataset_hash(ds2))

    def test_jsonl_round_trip(self):
        ds2 = loads_jsonl(dumps_jsonl(self.ds))
        self.assertEqual(dataset_hash(self.ds), dataset_hash(ds2))
        for split in SPLITS:
            for g in ds2[split]:
                self.assertEqual(validate_group(g), [])

    def test_split_properties(self):
        H = {tuple(p) for p in self.cfg["heldout_pairs"]}
        for g in self.ds["train"]:
            self.assertEqual(len(g.ops), 1)
            self.assertIn(g.para.template, (0, 1))
            for s in g.scenes:
                self.assertFalse(s.shape_color_pairs() & H)
        for g in self.ds["test_composition"]:
            self.assertEqual(len(g.ops), 2)
        for g in self.ds["test_heldout_template"]:
            self.assertEqual(g.para.template, 2)

    def test_audit_detects_scene_leak(self):
        ds = {k: list(v) for k, v in self.ds.items()}
        leaked = copy.copy(ds["train"][0])
        leaked.split = "test_iid"
        ds["test_iid"] = ds["test_iid"] + [leaked]
        rep = audit(ds, self.cfg)
        self.assertEqual(rep["checks"]["no_cross_split_scene"]["status"], "FAIL")
        self.assertEqual(rep["checks"]["no_cross_split_desc"]["status"], "FAIL")

    def test_audit_detects_paraphrase_leak(self):
        # Same caption content in another template must still count as a leak.
        ds = {k: list(v) for k, v in self.ds.items()}
        g = self.ds["train"][0]
        para = cap.Paraphrase(2, tuple(reversed(g.para.mention_order)), g.para.rel_pair, 1 - g.para.rel_dir)
        rng = random.Random(0)
        g2 = make_group(rng, g.base, g.ops, para, "leak", "test_heldout_template")
        self.assertEqual({k for k in group_keys(g2) if k[0] == "desc"},
                         {k for k in group_keys(g) if k[0] == "desc"})
        ds["test_heldout_template"] = ds["test_heldout_template"] + [g2]
        rep = audit(ds, self.cfg)
        self.assertEqual(rep["checks"]["no_cross_split_desc"]["status"], "FAIL")

    def test_audit_detects_heldout_pair_in_train(self):
        ds = {k: list(v) for k, v in self.ds.items()}
        g = next(x for x in self.ds["test_heldout_pairs"])
        leaked = copy.copy(g)
        leaked.split = "train"
        ds["train"] = ds["train"] + [leaked]
        rep = audit(ds, self.cfg)
        self.assertEqual(rep["checks"]["train:no_heldout_pairs"]["status"], "FAIL")

    def test_audit_detects_template_leak(self):
        ds = {k: list(v) for k, v in self.ds.items()}
        leaked = copy.copy(self.ds["test_heldout_template"][0])
        leaked.split = "dev"
        ds["dev"] = ds["dev"] + [leaked]
        rep = audit(ds, self.cfg)
        self.assertEqual(rep["checks"]["dev:templates_as_specified"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
