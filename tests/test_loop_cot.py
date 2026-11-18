import random
import unittest

import torch

from loop_jev.experiments.loop_cot_suite import SEED, TRAIN_COUNT, VALIDATION_COUNT, validation_rows
from loop_jev.training.loop_cot import IGNORED, round_targets, token_loss


class LoopCoTTest(unittest.TestCase):
    def test_shuffled_targets_only_change_round_assignment(self):
        tokens = list(range(90))
        ordered = round_targets(tokens, 8, order="ordered", row_id="id", seed=42)
        self.assertEqual([len(part) for part in ordered], [8, 8, 8])
        for i, part in enumerate(ordered):
            self.assertTrue(all(30 * i <= token < 30 * (i + 1) for token in part))
        assignments = [round_targets(tokens, 8, order="shuffled", row_id=str(i), seed=42)
                       for i in range(20)]
        for parts in assignments:
            self.assertNotEqual(parts, ordered)
            self.assertEqual(sorted(parts), ordered)
        self.assertGreater(len({tuple(part[0] for part in parts) for parts in assignments}), 1)


    def test_unlabeled_positions_do_not_affect_loss_or_gradient(self):
        targets = torch.tensor([round_targets([2], 2, order="ordered", row_id="id", seed=42)])
        logits = torch.randn(1, 6, 4, requires_grad=True)
        loss = token_loss(logits, targets)
        expected = -logits[0, 4].log_softmax(-1)[2]
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertFalse(logits.grad[0, [0, 1, 2, 3, 5]].any())
        self.assertEqual(token_loss(logits, torch.full_like(targets, IGNORED)).item(), 0)


    def test_validation_excludes_the_original_training_sample(self):
        rows = [{"id": i} for i in range(3000)]
        training_ids = set(random.Random(SEED).sample(range(len(rows)), TRAIN_COUNT))
        validation = validation_rows(rows)
        validation_ids = {row["id"] for row in validation}
        self.assertEqual(len(validation_ids), VALIDATION_COUNT)
        self.assertFalse(validation_ids & training_ids)
        self.assertEqual(validation, validation_rows(rows))
