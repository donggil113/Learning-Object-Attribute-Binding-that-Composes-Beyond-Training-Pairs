import random
import unittest

from bindcomp import vocab
from bindcomp.groups import check_invariants, diff_profile, expected_profile
from bindcomp.ops import (InvalidOp, applicable_ops, apply_ops, reachable_in_one, replace_attr,
                          replace_shape, swap_attr, swap_slots)
from bindcomp.scene import InvalidScene, Obj, Scene, sample_scene


def scene3():
    return Scene((
        Obj(0, "cube", "red", "metal", 0),
        Obj(1, "sphere", "blue", "rubber", 1),
        Obj(2, "cone", "green", "metal", 2),
    ))


class TestScene(unittest.TestCase):
    def test_rejects_invalid(self):
        with self.assertRaises(InvalidScene):
            Scene((Obj(0, "cube", "red", "metal", 0), Obj(1, "cube", "red", "metal", 1)))
        with self.assertRaises(InvalidScene):
            Scene((Obj(0, "cube", "red", "metal", 0), Obj(1, "cone", "red", "metal", 0)))
        with self.assertRaises(InvalidScene):
            Scene((Obj(0, "cube", "pink", "metal", 0),))

    def test_keys_ignore_oid(self):
        a = scene3()
        b = Scene(tuple(Obj(10 + o.oid, o.shape, o.color, o.material, o.slot) for o in a.objects))
        self.assertEqual(a.canonical_key(), b.canonical_key())
        self.assertEqual(a.object_set_key(), b.object_set_key())

    def test_sample_scene_respects_forbidden(self):
        rng = random.Random(0)
        H = frozenset(vocab.DEFAULT_HELDOUT_PAIRS)
        for _ in range(500):
            s = sample_scene(rng, 3, H)
            self.assertFalse(s.shape_color_pairs() & H)


class TestOps(unittest.TestCase):
    def test_binding_swap_preserves_multisets_changes_binding(self):
        s = scene3()
        t = swap_attr("color", 0, 1).apply(s)
        for f in ("shape", "color", "material", "slot"):
            self.assertEqual(s.multiset(f), t.multiset(f), f)
        self.assertNotEqual(s.bindings("color"), t.bindings("color"))
        self.assertEqual(s.bindings("material"), t.bindings("material"))
        self.assertEqual(t.obj(0).color, "blue")
        self.assertEqual(t.obj(1).color, "red")
        self.assertEqual(check_invariants(s, t, (swap_attr("color", 0, 1),)), [])

    def test_noop_swap_rejected(self):
        s = scene3()
        with self.assertRaises(InvalidOp):
            swap_attr("material", 0, 2).apply(s)  # both metal

    def test_degenerate_same_shape_swap_violates_contract(self):
        # Swapping colors between two cubes leaves the shape-color binding
        # multiset unchanged: it is not a binding change and must be rejected.
        s = Scene((Obj(0, "cube", "red", "metal", 0), Obj(1, "cube", "blue", "rubber", 1),
                   Obj(2, "cone", "green", "metal", 2)))
        op = swap_attr("color", 0, 1)
        t = op.apply(s)
        viol = check_invariants(s, t, (op,))
        self.assertTrue(any("bind_color_equal" in v for v in viol), viol)

    def test_replace_attr_changes_exactly_one(self):
        s = scene3()
        op = replace_attr("color", 2, "purple")
        t = op.apply(s)
        self.assertEqual(check_invariants(s, t, (op,)), [])
        self.assertEqual(s.multiset("shape"), t.multiset("shape"))
        self.assertEqual(sum((s.multiset("color") - t.multiset("color")).values()), 1)

    def test_replace_shape(self):
        s = scene3()
        op = replace_shape(1, "cylinder")
        t = op.apply(s)
        self.assertEqual(check_invariants(s, t, (op,)), [])
        self.assertEqual(s.multiset("color"), t.multiset("color"))
        self.assertNotEqual(s.multiset("shape"), t.multiset("shape"))

    def test_relation_swap_preserves_everything_but_layout(self):
        s = scene3()
        op = swap_slots(0, 2)
        t = op.apply(s)
        prof = diff_profile(s, t)
        self.assertTrue(prof["triples_equal"])
        self.assertTrue(prof["bind_color_equal"] and prof["bind_material_equal"])
        self.assertFalse(prof["oid_layout_equal"])
        self.assertEqual(check_invariants(s, t, (op,)), [])

    def test_involutions(self):
        rng = random.Random(1)
        for _ in range(300):
            s = sample_scene(rng, 3)
            for op in applicable_ops(s, ("binding", "relation")):
                self.assertEqual(op.apply(op.apply(s)).canonical_key(), s.canonical_key())

    def test_disjoint_ops_commute(self):
        rng = random.Random(2)
        n = 0
        for _ in range(200):
            s = sample_scene(rng, 3)
            ops = applicable_ops(s)
            for a in ops:
                for b in ops:
                    if a.touched_oids & b.touched_oids:
                        continue
                    try:
                        ab = apply_ops(s, (a, b))
                        ba = apply_ops(s, (b, a))
                    except InvalidOp:
                        continue
                    n += 1
                    self.assertEqual(ab.canonical_key(), ba.canonical_key())
        self.assertGreater(n, 100)

    def test_every_applicable_single_op_meets_or_flags_contract(self):
        # Each op either satisfies its expected profile or is flagged; never a
        # silent mismatch between recomputed profile and expectation.
        rng = random.Random(3)
        for _ in range(200):
            s = sample_scene(rng, 3)
            for op in applicable_ops(s):
                t = op.apply(s)
                prof = diff_profile(s, t)
                exp = expected_profile(op)
                viol = check_invariants(s, t, (op,))
                mismatch = [f for f, v in exp.items() if v is not None and prof[f] != v]
                self.assertEqual(bool(mismatch), bool(viol))

    def test_reachable_in_one(self):
        s = scene3()
        self.assertTrue(reachable_in_one(s, swap_attr("color", 0, 1).apply(s)))
        two = apply_ops(s, (swap_attr("color", 0, 1), replace_shape(2, "cube")))
        self.assertFalse(reachable_in_one(s, two))
        # Two swaps that cancel are reachable (identity is not a single op, so
        # check a composition that collapses to one swap instead).
        collapsed = apply_ops(s, (swap_slots(0, 1), swap_slots(0, 1), swap_slots(1, 2)))
        self.assertTrue(reachable_in_one(s, collapsed))


if __name__ == "__main__":
    unittest.main()
