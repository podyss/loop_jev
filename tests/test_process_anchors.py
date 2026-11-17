import unittest

import torch

from loop_jev.training.process_anchors import (
    make_process_prefixes, process_alignment_loss, split_reasoning_steps,
)


class ProcessAnchorsTest(unittest.TestCase):
    def test_prefix_does_not_include_later_steps(self):
        steps = ["Find a.", "Find b.", "Combine.", "Solve.", "Check."]
        prefixes = make_process_prefixes(steps, 3)
        self.assertEqual(prefixes[0], "Find a.\nFind b.")
        self.assertEqual(prefixes[1], "\n".join(steps[:4]))
        self.assertEqual(prefixes[2], "\n".join(steps))

    def test_short_and_empty_traces(self):
        self.assertEqual(make_process_prefixes(["a", "b"], 3), ["a", "a\nb", "a\nb"])
        self.assertEqual(make_process_prefixes([], 3), ["", "", ""])
        self.assertEqual(make_process_prefixes(["a", "b"], 1), ["a\nb"])

    def test_preserves_arithmetic_and_prefers_lines(self):
        text = "Price is 0.2. Then compute <<0.2*5=1>>1."
        self.assertEqual(split_reasoning_steps(text),
                         ["Price is 0.2.", "Then compute <<0.2*5=1>>1."])
        self.assertEqual(split_reasoning_steps(text + "\nTotal is 1."), [text, "Total is 1."])

    def test_loss_reaches_each_loop_but_not_teacher_or_masked_sample(self):
        students = [torch.tensor([[1., 0.], [1., 0.]], requires_grad=True) for _ in range(3)]
        teachers = [torch.tensor([[0., 1.], [0., 1.]], requires_grad=True) for _ in range(3)]
        loss = process_alignment_loss(students, teachers, torch.tensor([True, False]), .5)
        self.assertAlmostEqual(loss.item(), 1.)
        loss.backward()
        for student, teacher in zip(students, teachers):
            self.assertGreater(student.grad[0].abs().sum().item(), 0)
            self.assertEqual(student.grad[1].abs().sum().item(), 0)
            self.assertIsNone(teacher.grad)
        self.assertTrue(torch.allclose(students[0].grad, 2 * students[1].grad))

    def test_empty_batch_has_finite_zero_loss_and_gradient(self):
        student = torch.randn(2, 4, requires_grad=True)
        loss = process_alignment_loss([student], [torch.randn(2, 4)], torch.zeros(2, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(student.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
