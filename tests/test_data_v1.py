import json
import unittest
from pathlib import Path

from bindcomp.dataset import audit, build_dataset, dataset_hash, orbit_owner
from bindcomp.ops import applicable_ops
from bindcomp.scene import sample_scene

ROOT = Path(__file__).resolve().parents[1]
V0_SHA256 = "af7de8f60c227fe2b266ddc1ac226300b70366c4e26cb8e14593c95dd4a93068"


class TestDataV1(unittest.TestCase):
    def test_v0_dataset_unchanged(self):
        cfg = json.loads((ROOT / "configs" / "data_v0.json").read_text())
        ds, _ = build_dataset(cfg)
        self.assertEqual(dataset_hash(ds), V0_SHA256)

    def test_content_key_is_invariant_under_binding_necessary_edits(self):
        import random
        rng = random.Random(0)
        for _ in range(200):
            s = sample_scene(rng, 3)
            for op in applicable_ops(s):
                same = op.apply(s).content_key() == s.content_key()
                self.assertEqual(same, op.kind in ("binding", "relation"), op)

    def test_v1_small_orbit_split(self):
        cfg = json.loads((ROOT / "configs" / "data_v1.json").read_text())
        for split, spec in cfg["splits"].items():
            spec["n_groups"] = 40 if split == "train" else 12
        ds, _ = build_dataset(cfg)
        rep = audit(ds, cfg)
        self.assertEqual({k: v for k, v in rep["checks"].items() if v["status"] == "FAIL"}, {})
        self.assertIn("no_cross_split_orbit", rep["checks"])
        for split, gs in ds.items():
            for g in gs:
                for s in g.scenes:
                    self.assertEqual(orbit_owner(s.content_key(), cfg), split)


class TestManifest(unittest.TestCase):
    def test_run_outputs_do_not_mark_tree_dirty(self):
        from bindcomp.manifest import dirty_code_lines
        # The first porcelain line may start with a space (unstaged modification).
        status = " M runs/pixel_v1_cpu_ledger.json\n?? runs/x/log.txt\n M bindcomp/head.py\n"
        self.assertEqual(dirty_code_lines(status), [" M bindcomp/head.py"])


if __name__ == "__main__":
    unittest.main()
