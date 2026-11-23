import math
import unittest

import torch

from loop_jev.training.bidirectional_arithmetic import arithmetic_loss, compile_targets


class BidirectionalArithmeticTests(unittest.TestCase):
    def test_exact_forward_and_backward_values_follow_option_order(self):
        row = dict(question="12 dollars for 5 meters; find the cost of 3 meters.",
                   rationale="<<12/5=2.4>> <<2.4*3=7.2>>", options=[6, 7.2, 9, 12], gold_index=1)
        targets = compile_targets(row)
        torch.testing.assert_close(torch.tensor(targets[:5]).sinh(), torch.tensor([2.4, 2., 2.4, 3., 4.]))
        self.assertEqual(min(range(4), key=lambda i: targets[5 + i]), 1)
        self.assertEqual(targets[6], 0.)
        row["options"][0] = 7.2
        self.assertIsNone(compile_targets(row))

    def test_consistency_reaches_all_rounds_and_masks_missing_targets(self):
        values = [math.asinh(x) for x in (2.4, 2, 2.4, 3, 4)]
        labels = values + [abs(values[0] - x) for x in values[1:]]
        targets = torch.tensor([labels, labels])
        predictions = targets.clone()
        predictions[0, 5:9] += .5
        predictions.requires_grad_()
        loss, stats = arithmetic_loss(predictions, targets, torch.tensor([True, False]))
        self.assertGreater(stats[1].item(), 0.)
        loss.backward()
        self.assertTrue(predictions.grad[0, :5].abs().sum().item() > 0)
        self.assertTrue(predictions.grad[0, 5:].abs().sum().item() > 0)
        self.assertTrue(predictions.grad[1].eq(0).all())
