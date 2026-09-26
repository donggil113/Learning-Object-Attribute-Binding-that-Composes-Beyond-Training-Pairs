"""Pixel-path regression tests (require the repo .venv: numpy, torch, Pillow, open_clip)."""

import importlib.util
import json
import random
import unittest
from pathlib import Path

HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "numpy", "PIL", "open_clip"))
ROOT = Path(__file__).resolve().parents[1]

if HAS_TORCH:
    import numpy as np
    import torch

    from bindcomp import pixel_render
    from bindcomp.head import FactorizedHead
    from bindcomp.losses import edit_consistency as ref_edit
    from bindcomp.losses import info_nce as ref_info_nce
    from bindcomp.pixel import (ClipImageOnlyScorer, FeatGroup, ZeroShotClipScorer, shuffled)
    from bindcomp.scene import sample_scene
    from bindcomp.shortcuts import OracleScorer, TextOnlyScorer
    from bindcomp.torch_head import (DTYPE, TorchFactorizedHead, TorchHeadScorer, batch_loss,
                                     edit_consistency, info_nce, pad)
    from bindcomp.train import VARIANTS, batch_forward
    from tests.test_head_losses import make_enc_groups


class Poison:
    """Stands in for group metadata; any attribute access fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"model scorer read metadata field {name!r}")


def to_feat(eg, meta=None):
    t = lambda v: torch.tensor(v, dtype=DTYPE)  # noqa: E731
    img = torch.stack([t(eg.img[0]), t(eg.img[1])])
    return FeatGroup(eg.gid, eg.kind_label, eg.same_words, img, img.mean(1), [t(x) for x in eg.txt],
                     [t(x) for x in eg.txt_hp], torch.stack([t(x).mean(0) for x in eg.txt]),
                     torch.stack([t(x).mean(0) for x in eg.txt_hp]), tuple(eg.group.captions), eg.captions_hp,
                     eg.group if meta is None else meta)


@unittest.skipUnless(HAS_TORCH, "requires the repo .venv (torch)")
class TestRenderer(unittest.TestCase):
    def test_deterministic_and_seed_dependent(self):
        s = sample_scene(random.Random(0), 3)
        a = np.asarray(pixel_render.render(s, 123))
        self.assertTrue(np.array_equal(a, np.asarray(pixel_render.render(s, 123))))
        self.assertFalse(np.array_equal(a, np.asarray(pixel_render.render(s, 124))))

    def test_pixels_encode_every_object_and_slot(self):
        rng = random.Random(7)
        for _ in range(200):
            s = sample_scene(rng, 3)
            self.assertEqual(pixel_render.decode(pixel_render.render(s, rng.getrandbits(48))),
                             pixel_render.scene_signature(s))

    def test_unsupported_values_fail_loudly(self):
        with self.assertRaises(ValueError):
            pixel_render._face_masks("pyramid", 100, 100, 20)
        with self.assertRaises(ValueError):
            pixel_render._shade_field("cube", "glass", 100, 100, 20)


@unittest.skipUnless(HAS_TORCH, "requires the repo .venv (torch)")
class TestTorchBridge(unittest.TestCase):
    def test_forward_matches_reference_head(self):
        groups = make_enc_groups(4, seed=3)
        ref = FactorizedHead(d=24, hc=3, hb=4, window=5, seed=9)
        th = TorchFactorizedHead(d=24, hc=3, hb=4, window=5, seed=9)
        for eg in groups:
            for m in (0, 1):
                c, b, _ = ref.encode_image(eg.img[m])
                tc, tb = th.encode_image(torch.tensor([eg.img[m]], dtype=DTYPE))
                self.assertTrue(torch.allclose(tc[0], torch.tensor(c, dtype=DTYPE), atol=1e-10))
                self.assertTrue(torch.allclose(tb[0], torch.tensor(b, dtype=DTYPE), atol=1e-10))
        txts = [eg.txt[m] for eg in groups for m in (0, 1)]
        toks, lengths = pad([torch.tensor(t, dtype=DTYPE) for t in txts])
        tc, tb = th.encode_text(toks, lengths)
        for k, t in enumerate(txts):
            c, b, _ = ref.encode_text(t)
            self.assertTrue(torch.allclose(tc[k], torch.tensor(c, dtype=DTYPE), atol=1e-10))
            self.assertTrue(torch.allclose(tb[k], torch.tensor(b, dtype=DTYPE), atol=1e-10))

    def test_losses_match_reference(self):
        rng = random.Random(0)
        S = [[rng.gauss(0, 1) for _ in range(6)] for _ in range(6)]
        allowed = [[rng.random() < 0.6 and i != j for j in range(6)] for i in range(6)]
        self.assertAlmostEqual(float(info_nce(torch.tensor(S, dtype=DTYPE), torch.tensor(allowed), 1.7)),
                               ref_info_nce(S, allowed, 1.7)[0], places=10)
        DI = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(4)]
        DT = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(4)]
        self.assertAlmostEqual(float(edit_consistency(torch.tensor(DI, dtype=DTYPE), torch.tensor(DT, dtype=DTYPE), 0.3)),
                               ref_edit(DI, DT, tau=0.3)[0], places=10)

    def test_gradients_match_reference_training_step(self):
        groups = make_enc_groups(3, seed=11)
        feats = [to_feat(eg) for eg in groups]
        for variant, lam in (("hardneg", 0.0), ("hardneg_eq", 0.5), ("inbatch", 0.0)):
            ref = FactorizedHead(d=24, hc=3, hb=3, window=2, seed=1)
            th = TorchFactorizedHead(d=24, hc=3, hb=3, window=2, seed=1)
            out, grads = batch_forward(ref, groups, VARIANTS[variant]["hard_neg"], lam, scale=2.0, tau_eq=0.5)
            loss, _, _ = batch_loss(th, feats, VARIANTS[variant]["hard_neg"], lam, scale=2.0, tau_eq=0.5)
            loss.backward()
            self.assertAlmostEqual(float(loss.detach()), out["loss"], places=10)
            for k, p in th.p.items():
                self.assertTrue(torch.allclose(p.grad, torch.tensor(grads[k], dtype=DTYPE), atol=1e-10), (variant, k))


@unittest.skipUnless(HAS_TORCH, "requires the repo .venv (torch)")
class TestProvenanceGuard(unittest.TestCase):
    def test_model_scorers_never_read_metadata(self):
        clean = [to_feat(eg) for eg in make_enc_groups(8, seed=5)]
        poisoned = [to_feat(eg, meta=Poison()) for eg in make_enc_groups(8, seed=5)]
        head = TorchFactorizedHead(d=24, seed=0)
        scorers = [TorchHeadScorer(head, ch) for ch in ("all", "content", "binding")]
        scorers += [ZeroShotClipScorer(), ClipImageOnlyScorer().fit(clean), TextOnlyScorer().fit(clean)]
        for sc in scorers:
            for fg in poisoned + shuffled(poisoned, "image") + shuffled(poisoned, "text"):
                S, S_hp = sc.score_group(fg)
                self.assertEqual(len(S), 2)
        with self.assertRaises(AssertionError):
            OracleScorer().score_group(poisoned[0])  # the oracle control does read metadata


@unittest.skipUnless(HAS_TORCH and (ROOT / ".cache/models/clip_vit_b32_laion2b/open_clip_model.safetensors").exists(),
                     "requires the pinned encoder checkpoint")
class TestClipFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bindcomp.clip_features import ClipEncoder
        cfg = json.loads((ROOT / "configs" / "encoder_pixel_v1.json").read_text())
        cls.enc = ClipEncoder(cfg, ROOT)

    def test_tokens_consistent_with_open_clip_pooled_outputs(self):
        s = sample_scene(random.Random(1), 3)
        ims = [pixel_render.render(s, 5), pixel_render.render(s, 6)]
        pooled, toks = self.enc.encode_images(ims)
        self.assertEqual(tuple(toks.shape), (2, 50, 512))
        m = self.enc.model
        m.visual.output_tokens = False
        ref = m.encode_image(torch.stack([self.enc.preprocess(im) for im in ims]))
        m.visual.output_tokens = True
        self.assertTrue(torch.allclose(pooled, ref, atol=1e-5))
        self.assertTrue(torch.allclose(toks[:, 0], pooled, atol=1e-6))
        texts = ["a red metal cube , a blue rubber sphere", "a cube that is red and metal"]
        tp, tt = self.enc.encode_texts(texts)
        self.assertTrue(torch.allclose(tp, m.encode_text(self.enc.tokenizer(texts)), atol=1e-5))
        self.assertTrue(torch.allclose(tt[0][-1], tp[0]))
        again, _ = self.enc.encode_images(ims)
        self.assertTrue(torch.equal(again, pooled))

    def test_rejects_non_pixel_non_text_inputs(self):
        s = sample_scene(random.Random(2), 3)
        with self.assertRaises(TypeError):
            self.enc.encode_images([s])
        with self.assertRaises(TypeError):
            self.enc.encode_texts([("a", "red", "cube")])


if __name__ == "__main__":
    unittest.main()
