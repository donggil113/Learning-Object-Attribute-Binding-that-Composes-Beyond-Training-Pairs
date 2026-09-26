import random
import unittest

from bindcomp import captions as cap
from bindcomp.oracle import satisfies
from bindcomp.ops import replace_attr, replace_shape, swap_attr, swap_slots
from bindcomp.scene import Obj, Scene, sample_scene


def scene3():
    return Scene((
        Obj(0, "cube", "red", "metal", 0),
        Obj(1, "sphere", "blue", "rubber", 1),
        Obj(2, "cone", "green", "metal", 2),
    ))


def partial(*objs, left_of=()):
    return cap.Desc(tuple(objs), tuple(left_of))


class TestRealizeParse(unittest.TestCase):
    def test_round_trip_all_templates(self):
        rng = random.Random(0)
        for _ in range(300):
            s = sample_scene(rng, 3)
            for t in cap.TEMPLATES:
                para = cap.sample_paraphrase(rng, s, [t])
                d = cap.describe(s, para)
                toks = cap.realize(d, t, para.rel_dir)
                self.assertEqual(cap.parse(toks, t).canonical(), d.canonical())

    def test_example_strings(self):
        s = scene3()
        para = cap.Paraphrase(0, (1, 0, 2), (0, 2), 0)
        toks = cap.realize(cap.describe(s, para), 0, 0)
        self.assertEqual(" ".join(toks),
                         "a blue rubber sphere , a red metal cube , a green metal cone ; "
                         "the red metal cube is left of the green metal cone")
        para1 = cap.Paraphrase(1, (0, 1, 2), (0, 2), 1)
        toks = cap.realize(cap.describe(s, para1), 1, 1)
        self.assertTrue(" ".join(toks).endswith(
            "the cone that is green and metal is right of the cube that is red and metal"))

    def test_converse_direction_same_meaning(self):
        s = scene3()
        for t in cap.TEMPLATES:
            d = cap.describe(s, cap.Paraphrase(t, (0, 1, 2), (0, 1), 0))
            a = cap.parse(cap.realize(d, t, 0), t)
            b = cap.parse(cap.realize(d, t, 1), t)
            self.assertEqual(a.canonical(), b.canonical())


class TestOracle(unittest.TestCase):
    def test_full_description_true(self):
        s = scene3()
        d = cap.describe(s, cap.Paraphrase(0, (2, 0, 1), (1, 2), 0))
        self.assertTrue(satisfies(s, d))

    def test_binding_swap_flips_truth(self):
        s = scene3()
        t = swap_attr("color", 0, 1).apply(s)
        para = cap.Paraphrase(0, (0, 1, 2), (0, 2), 0)
        d_s, d_t = cap.describe(s, para), cap.describe(t, para)
        self.assertEqual([[satisfies(x, y) for y in (d_s, d_t)] for x in (s, t)],
                         [[True, False], [False, True]])

    def test_partial_captions_track_exactly_what_changed(self):
        s = scene3()
        t = swap_attr("color", 0, 1).apply(s)
        untouched = partial(("cone", "green", "metal"))
        self.assertTrue(satisfies(s, untouched) and satisfies(t, untouched))
        # Mentioning only colors or only shapes cannot see a binding swap.
        colors_only = partial((None, "red", None), (None, "blue", None))
        self.assertTrue(satisfies(s, colors_only) and satisfies(t, colors_only))
        shapes_only = partial(("cube", None, None), ("sphere", None, None))
        self.assertTrue(satisfies(s, shapes_only) and satisfies(t, shapes_only))
        # A bound phrase does.
        red_cube = partial(("cube", "red", None))
        self.assertTrue(satisfies(s, red_cube))
        self.assertFalse(satisfies(t, red_cube))

    def test_attribute_and_object_changes(self):
        s = scene3()
        t = replace_attr("material", 1, "metal").apply(s)
        self.assertFalse(satisfies(t, partial(("sphere", "blue", "rubber"))))
        self.assertTrue(satisfies(t, partial(("sphere", "blue", "metal"))))
        u = replace_shape(2, "cylinder").apply(s)
        self.assertFalse(satisfies(u, partial(("cone", None, None))))
        self.assertTrue(satisfies(u, partial(("cylinder", "green", "metal"))))

    def test_relation_change(self):
        s = scene3()
        t = swap_slots(0, 2).apply(s)
        rel = partial(("cube", None, None), ("cone", None, None), left_of=((0, 1),))
        self.assertTrue(satisfies(s, rel))
        self.assertFalse(satisfies(t, rel))
        no_rel = partial(("cube", "red", "metal"), ("cone", "green", "metal"))
        self.assertTrue(satisfies(s, no_rel) and satisfies(t, no_rel))
        # Relation between untouched pair is preserved only if order preserved.
        t2 = swap_slots(0, 1).apply(s)  # sphere now at 0, cube at 1, cone at 2
        cube_left_of_cone = partial(("cube", None, None), ("cone", None, None), left_of=((0, 1),))
        self.assertTrue(satisfies(t2, cube_left_of_cone))

    def test_injective_assignment(self):
        s = scene3()
        two_metal = partial((None, None, "metal"), (None, None, "metal"))
        three_metal = partial((None, None, "metal"), (None, None, "metal"), (None, None, "metal"))
        self.assertTrue(satisfies(s, two_metal))
        self.assertFalse(satisfies(s, three_metal))


if __name__ == "__main__":
    unittest.main()
