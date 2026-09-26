import unittest

from bindcomp.evaluate import (auc, evaluate_groups, fit_threshold, full_report, group_outcome,
                               pair_scores, summarize)
from bindcomp.head import FactorizedHead
from bindcomp.shortcuts import (HeadScorer, ImageOnlyScorer, OracleScorer, RandomScorer,
                                TextOnlyScorer, blind_detectability)
from tests.test_head_losses import make_enc_groups


class TestMetrics(unittest.TestCase):
    def test_group_outcome_definitions(self):
        # Correct on text (each image prefers its caption), wrong on image
        # (caption 0 prefers image 1).
        S = [[0.9, 0.5], [0.95, 0.96]]
        r = group_outcome(S, [0.9, 0.7])
        self.assertEqual((r["text"], r["image"], r["group"]), (1.0, 0.0, 0.0))
        self.assertEqual(r["match"], 1.0)  # 0.9 + 0.96 > 0.5 + 0.95
        wrong_text = group_outcome([[0.9, 0.5], [0.95, 0.6]])
        self.assertEqual(wrong_text["text"], 0.0)
        ties = group_outcome([[1.0, 1.0], [1.0, 1.0]], [1.0, 1.0])
        self.assertEqual(sum(ties.values()), 0.0)
        noisy = group_outcome([[1.0, 1.0 - 1e-15], [1.0, 1.0 + 1e-15]], [1.0, 1.0])
        self.assertEqual(sum(noisy.values()), 0.0)  # round-off is a tie, not a win

    def test_auc(self):
        self.assertEqual(auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
        self.assertEqual(auc([1, 1, 1, 1], [0, 1, 0, 1]), 0.5)

    def test_fit_threshold(self):
        thr, acc = fit_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
        self.assertEqual(acc, 1.0)
        self.assertTrue(0.2 < thr < 0.8)


class TestScorers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = make_enc_groups(40, seed=31)
        cls.train_groups = make_enc_groups(80, seed=32)

    def test_random_scorer_hits_chance(self):
        rows = evaluate_groups(RandomScorer(seed=0), self.groups * 150)  # 6000 draws
        s = summarize(rows)["all"]
        self.assertAlmostEqual(s["text"], 1 / 4, delta=0.02)
        self.assertAlmostEqual(s["image"], 1 / 4, delta=0.02)
        self.assertAlmostEqual(s["group"], 1 / 6, delta=0.02)
        self.assertAlmostEqual(s["match"], 1 / 2, delta=0.02)
        self.assertAlmostEqual(s["aug"], 1 / 3, delta=0.02)

    def test_oracle_is_perfect(self):
        rows = evaluate_groups(OracleScorer(), self.groups)
        s = summarize(rows)["all"]
        for m in ("text", "image", "group", "match", "aug", "aug_group"):
            self.assertEqual(s[m], 1.0, m)
        sc, lab = pair_scores(rows, self.groups)
        self.assertEqual(auc(sc, lab), 1.0)

    def test_blind_scorers_score_zero_on_group_metrics(self):
        for scorer in (TextOnlyScorer().fit(self.train_groups), ImageOnlyScorer().fit(self.train_groups)):
            s = summarize(evaluate_groups(scorer, self.groups))["all"]
            for m in ("text", "image", "group", "match", "aug_group"):
                self.assertEqual(s[m], 0.0, (scorer.name, m))

    def test_content_channel_cannot_solve_same_word_binding_groups(self):
        head = FactorizedHead(d=24, seed=0)
        groups = [g for g in make_enc_groups(80, seed=33, kinds=("binding", "relation")) if g.same_words]
        self.assertGreater(len(groups), 10)
        s = summarize(evaluate_groups(HeadScorer(head, "content"), groups))["all"]
        # c_T identical for both captions -> every text comparison ties.
        self.assertEqual(s["text"], 0.0)
        self.assertEqual(s["match"], 0.0)

    def test_full_report_runs(self):
        head = FactorizedHead(d=24, seed=0)
        rep = full_report(HeadScorer(head), {"calib": self.groups[:20], "test": self.groups[20:]}, n_boot=50)
        self.assertIn("pair_acc_at_calib_threshold", rep["test"])
        self.assertIsNotNone(rep["calib_threshold"])

    def test_blind_detectability_runs(self):
        for modality in ("text", "image"):
            r = blind_detectability(self.train_groups, self.groups, modality)
            self.assertTrue(0.0 <= r["auc"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
