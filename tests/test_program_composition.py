import unittest
from fractions import Fraction

import torch

from loop_jev.training.program_composition import compile_program, program_loss


def row(question, rationale, answer):
    return dict(question=question, rationale=rationale, options=[answer], gold_index=0)


class ProgramCompositionTest(unittest.TestCase):
    def test_compile_fractional_chain_and_evaluate_new_input(self):
        program = compile_program(row("A material costs 12 dollars for 5 meters.",
                                      "<<12/5=2.4>> <<2.4*3=7.2>>", "7.2"))
        self.assertIsNotNone(program)
        upstream, downstream, whole = program.functions
        self.assertEqual(whole, (Fraction(3, 5), 0))
        x = Fraction(7)
        evaluate = lambda fn, value: fn[0] * value + fn[1]
        self.assertEqual(evaluate(downstream, evaluate(upstream, x)), evaluate(whole, x))
        self.assertEqual(evaluate(whole, x), Fraction(21, 5))

    def test_reject_bad_or_ambiguous_grounding(self):
        for trace, answer in (("<<12/5=2.5>><<2.5*3=7.5>>", "7.5"),
                              ("<<12/5=2.4>><<2.4*3=7.2>>", "8"),
                              ("<<12/5=2.4>><<2.4*3=7.2>>", "7.2")):
            # The question also names 2.4; the compiler cannot know what the second expression refers to.
            self.assertIsNone(compile_program(row("12 for 5 with a 2.4 service charge", trace, answer)))

    def test_consistency_gradient_and_missing_targets(self):
        targets = torch.tensor([[[2., 4.], [3., -2.], [6., 10.]]] * 2).asinh()
        valid = torch.tensor([True, False])
        _, stats = program_loss(targets, targets, valid)
        self.assertLess(stats[1].item(), 1e-10)
        predictions = targets.clone()
        predictions[0, 1, 0] += 0.5
        predictions.requires_grad_()
        loss, stats = program_loss(predictions, targets, valid)
        self.assertGreater(stats[1].item(), 0)
        self.assertEqual(stats[2].item(), 1)
        loss.backward()
        self.assertTrue(predictions.grad[0].abs().sum(-1).gt(0).all())
        self.assertTrue(predictions.grad[1].eq(0).all())


if __name__ == "__main__":
    unittest.main()
