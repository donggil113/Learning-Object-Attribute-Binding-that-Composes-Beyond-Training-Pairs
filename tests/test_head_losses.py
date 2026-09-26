import random
import unittest
from math import log

from bindcomp import captions as cap
from bindcomp.collapse import collapse_flags, embedding_stats, head_report
from bindcomp.encode import encode_group
from bindcomp.groups import make_group
from bindcomp.head import FactorizedHead
from bindcomp.losses import delta_mse, edit_consistency, info_nce
from bindcomp.ops import applicable_ops, apply_ops, sample_op
from bindcomp.render import ProxyEncoder
from bindcomp.scene import sample_scene
from bindcomp.train import VARIANTS, batch_forward, train

ENC = ProxyEncoder(dim=24, seed=0)


def make_enc_groups(n, seed=0, kinds=("binding", "attribute", "object", "relation")):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = sample_scene(rng, 3)
        kind = kinds[len(out) % len(kinds)]
        try:
            op = sample_op(rng, s, kind)
            para = cap.sample_paraphrase(rng, s, [0, 1])
            g = make_group(rng, s, (op,), para, f"t{len(out)}", "train")
        except ValueError:
            continue
        out.append(encode_group(g, ENC))
    return out


def get_param(p, key):
    name, idx = key
    return p[name][idx[0]][idx[1]] if len(idx) == 2 else p[name][idx[0]]


def set_param(p, key, val):
    name, idx = key
    if len(idx) == 2:
        p[name][idx[0]][idx[1]] = val
    else:
        p[name][idx[0]] = val


class TestGradients(unittest.TestCase):
    def _gradcheck(self, variant, lam_eq=0.5):
        head = FactorizedHead(d=24, hc=3, hb=3, window=2, seed=1)
        batch = make_enc_groups(3, seed=11)
        spec = dict(VARIANTS[variant])
        lam = lam_eq if spec["lam_eq"] is None else spec["lam_eq"]
        out, grads = batch_forward(head, batch, spec["hard_neg"], lam, scale=2.0, tau_eq=0.5)
        rng = random.Random(0)
        keys = []
        for name, v in head.p.items():
            for _ in range(4):
                if isinstance(v[0], list):
                    keys.append((name, (rng.randrange(len(v)), rng.randrange(len(v[0])))))
                else:
                    keys.append((name, (rng.randrange(len(v)),)))
        eps = 1e-6
        for key in keys:
            x0 = get_param(head.p, key)
            set_param(head.p, key, x0 + eps)
            lp = batch_forward(head, batch, spec["hard_neg"], lam, 2.0, 0.5, want_grads=False)[0]["loss"]
            set_param(head.p, key, x0 - eps)
            lm = batch_forward(head, batch, spec["hard_neg"], lam, 2.0, 0.5, want_grads=False)[0]["loss"]
            set_param(head.p, key, x0)
            num = (lp - lm) / (2 * eps)
            ana = get_param(grads, key)
            self.assertAlmostEqual(num, ana, delta=1e-6 + 1e-4 * abs(num), msg=f"{variant} {key}")

    def test_gradcheck_inbatch(self):
        self._gradcheck("inbatch")

    def test_gradcheck_hardneg(self):
        self._gradcheck("hardneg")

    def test_gradcheck_hardneg_eq(self):
        self._gradcheck("hardneg_eq")


class TestHeadStructure(unittest.TestCase):
    def test_content_channel_is_binding_invariant(self):
        head = FactorizedHead(d=24, seed=2)
        rng = random.Random(3)
        n = 0
        for _ in range(100):
            s = sample_scene(rng, 3)
            for op in applicable_ops(s, ("binding", "relation")):
                t = op.apply(s)
                seed = rng.getrandbits(32)
                c0 = head.encode_image(ENC.image_tokens(s, seed))[0]
                c1 = head.encode_image(ENC.image_tokens(t, seed))[0]
                self.assertLess(max(abs(a - b) for a, b in zip(c0, c1)), 1e-9)
                n += 1
        self.assertGreater(n, 200)

    def test_text_content_invariant_to_word_order(self):
        head = FactorizedHead(d=24, seed=2)
        toks = "a red metal cube , a blue rubber sphere".split()
        perm = "a blue metal cube , a red rubber sphere".split()
        c0 = head.encode_text(ENC.text_tokens(toks))[0]
        c1 = head.encode_text(ENC.text_tokens(perm))[0]
        self.assertLess(max(abs(a - b) for a, b in zip(c0, c1)), 1e-9)
        b0 = head.encode_text(ENC.text_tokens(toks))[1]
        b1 = head.encode_text(ENC.text_tokens(perm))[1]
        self.assertGreater(max(abs(a - b) for a, b in zip(b0, b1)), 1e-6)

    def test_image_binding_additive_over_disjoint_edits(self):
        head = FactorizedHead(d=24, seed=4)
        rng = random.Random(5)
        checked = 0
        for _ in range(60):
            s = sample_scene(rng, 3)
            ops = applicable_ops(s)
            rng.shuffle(ops)
            for a in ops[:10]:
                for b in ops[:10]:
                    if a.touched_oids & b.touched_oids:
                        continue
                    try:
                        ab = apply_ops(s, (a, b))
                    except ValueError:
                        continue
                    seed = 17
                    bs = head.encode_image(ENC.image_tokens(s, seed))[1]
                    ba = head.encode_image(ENC.image_tokens(a.apply(s), seed))[1]
                    bb = head.encode_image(ENC.image_tokens(b.apply(s), seed))[1]
                    bab = head.encode_image(ENC.image_tokens(ab, seed))[1]
                    for x0, xa, xb, xab in zip(bs, ba, bb, bab):
                        self.assertAlmostEqual(xab - x0, (xa - x0) + (xb - x0), places=10)
                    checked += 1
        self.assertGreater(checked, 50)


class TestLossesAndCollapse(unittest.TestCase):
    def test_zero_solution_behaviour(self):
        B, d = 4, 5
        zeros = [[0.0] * d for _ in range(B)]
        l_eq = edit_consistency(zeros, zeros, tau=0.1)[0]
        self.assertAlmostEqual(l_eq, log(B), places=9)  # zero is NOT a minimizer
        self.assertEqual(delta_mse(zeros, zeros)[0], 0.0)  # naive version IS minimized by zero

    def test_edit_consistency_prefers_aligned_edits(self):
        rng = random.Random(0)
        DI = [[rng.gauss(0, 1) for _ in range(6)] for _ in range(5)]
        aligned = edit_consistency(DI, [list(v) for v in DI], tau=0.1)[0]
        shuffled = edit_consistency(DI, DI[1:] + DI[:1], tau=0.1)[0]
        self.assertLess(aligned, shuffled)

    def test_info_nce_mask_changes_denominator(self):
        S = [[2.0, 1.9], [1.9, 2.0]]
        l_hard = info_nce(S, [[False, True], [True, False]])[0]
        l_masked = info_nce(S, [[False, False], [False, False]])[0]
        self.assertGreater(l_hard, 0.5)
        self.assertAlmostEqual(l_masked, 0.0, places=12)

    def test_constant_and_zero_detection(self):
        self.assertIn("ZERO", collapse_flags(embedding_stats([[0.0] * 4] * 10)))
        self.assertIn("CONSTANT", collapse_flags(embedding_stats([[1.0, -2.0, 0.5, 3.0]] * 10)))
        rng = random.Random(0)
        rand = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(50)]
        self.assertEqual(collapse_flags(embedding_stats(rand)), [])

    def test_head_report_flags(self):
        groups = make_enc_groups(24, seed=7)
        head = FactorizedHead(d=24, seed=8)
        rep = head_report(head, groups)
        self.assertEqual([f for f in rep["flags"] if "LOW_RANK" not in f], [])
        dead = FactorizedHead(d=24, seed=8)
        for name in dead.p:
            if name != "alpha":
                dead.p[name] = [[0.0] * len(r) for r in dead.p[name]]
        rep = head_report(dead, groups)
        for f in ("c_img:ZERO", "b_img:ZERO", "c_txt:ZERO", "b_txt:ZERO",
                  "b_img:BINDING_INSENSITIVE", "b_txt:BINDING_INSENSITIVE"):
            self.assertIn(f, rep["flags"])
        # Binding channel dead, content alive -> binding insensitivity only.
        half = FactorizedHead(d=24, seed=8)
        half.p["A_img"] = [[0.0] * 24 for _ in half.p["A_img"]]
        rep = head_report(half, groups)
        self.assertIn("b_img:BINDING_INSENSITIVE", rep["flags"])
        self.assertNotIn("c_img:ZERO", rep["flags"])


class TestTrainSmoke(unittest.TestCase):
    def test_overfits_single_batch(self):
        groups = make_enc_groups(4, seed=21)
        for variant in ("inbatch", "hardneg", "hardneg_eq"):
            head = FactorizedHead(d=24, hc=4, hb=6, window=5, seed=3)
            tcfg = dict(lr=0.05, weight_decay=0.0, steps=40, batch_groups=4, logit_scale=1.0,
                        tau_eq=0.1, lam_eq=0.3, log_every=1000)
            hist = train(head, groups, tcfg, variant, seed=0)
            self.assertLess(hist[-1]["task"], 0.7 * hist[0]["task"], variant)


if __name__ == "__main__":
    unittest.main()
