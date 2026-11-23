import random
import unittest

import torch

from loop_jev.training.condition_pairs import make_pair, pair_groups, paired_margin_loss


class ConditionPairTests(unittest.TestCase):
    def test_condition_is_recomputed_and_both_rows_share_options(self):
        row = dict(id="natalia", question="Natalia sold 48 clips in April, half as many in May, and triple May's sales in June. How many clips did she sell in June?",
                   rationale="<<48/2=24>> then <<24*3=72>>", options=["30040", "84", "72", "96"], gold_index=2)
        original, changed = make_pair(row, random.Random(0))
        self.assertEqual(original["options"], changed["options"])
        amount = int(changed["question"].split()[2])
        self.assertNotEqual(amount, 48)
        self.assertEqual(changed["options"][changed["gold_index"]], str(3 * (amount // 2)))
        self.assertEqual(pair_groups([original, row, changed, row]), [[0, 2], [1, 3]])
        row["rationale"] = "An unannotated computation: 48/2=24.\n" + row["rationale"]
        self.assertIsNone(make_pair(row, random.Random(0)))

    def test_penalty_prefers_reversal_and_ignores_single_examples(self):
        logits = torch.tensor([[3., 0.], [3., 0.], [3., 0.], [0., 3.]], requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1])
        valid = torch.tensor([True, False])
        loss, stats = paired_margin_loss(logits, targets, valid)
        reversed_loss, reversed_stats = paired_margin_loss(logits[[0, 3, 2, 1]], targets, valid)
        self.assertGreater(loss.item(), reversed_loss.item())
        self.assertEqual((stats[1].item(), reversed_stats[1].item()), (0, 1))
        loss.backward()
        self.assertEqual(logits.grad[2:].abs().sum().item(), 0)
