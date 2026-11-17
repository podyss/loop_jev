import copy
import unittest

import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from loop_jev.looping import enable_looping, without_looping


class JointLoopNonloopTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)
        config = Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=4, num_attention_heads=2,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.0,
            use_cache=False,
        )
        self.student = Qwen3ForCausalLM(config).train()
        self.reference = copy.deepcopy(self.student)
        enable_looping(self.student, 1, 3, 3, damping=True, collect_all_readouts=False)
        enable_looping(self.reference, 1, 3, 3, damping=True, collect_all_readouts=False)
        self.student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        self.reference.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        self.ids = torch.tensor([[1, 3, 4, 7], [2, 9, 6, 5]])
        self.targets = torch.tensor([7, 8])
        self.teacher_logits = torch.randn(2, 32)

    def objective(self, logits):
        ce = F.cross_entropy(logits[:, :10], self.targets)
        kl = F.kl_div(F.log_softmax(logits / 2, -1),
                      F.softmax(self.teacher_logits / 2, -1),
                      reduction="batchmean") * 4
        return ce + kl

    def test_temporary_nonloop_forward_restores_loop_and_gradients(self):
        plain = Qwen3ForCausalLM(self.student.config).eval()
        plain.load_state_dict(self.student.state_dict())
        with torch.no_grad():
            looped = self.student(self.ids, logits_to_keep=1).logits[:, -1, :].float()
            expected_plain = plain(self.ids, logits_to_keep=1).logits[:, -1, :].float()
            with without_looping(self.student):
                direct = self.student(self.ids, logits_to_keep=1).logits[:, -1, :].float()
        torch.testing.assert_close(direct, expected_plain)
        torch.testing.assert_close(self.student(self.ids, logits_to_keep=1).logits[:, -1, :], looped)

        # Compare sequential backward to one combined graph at the same weights.
        self.student.zero_grad(set_to_none=True)
        self.reference.zero_grad(set_to_none=True)
        (0.5 * self.objective(
            self.student(self.ids, logits_to_keep=1).logits[:, -1, :].float()
        )).backward()
        self.student.model.loop_hidden_states = []
        with without_looping(self.student):
            direct_loss = self.objective(
                self.student(self.ids, logits_to_keep=1).logits[:, -1, :].float()
            )
        (0.5 * direct_loss).backward()
        loop_loss = self.objective(
            self.reference(self.ids, logits_to_keep=1).logits[:, -1, :].float()
        )
        with without_looping(self.reference):
            direct_reference = self.objective(
                self.reference(self.ids, logits_to_keep=1).logits[:, -1, :].float()
            )
        (0.5 * loop_loss + 0.5 * direct_reference).backward()
        for student_param, reference_param in zip(self.student.parameters(), self.reference.parameters()):
            torch.testing.assert_close(student_param.grad, reference_param.grad, atol=1e-5, rtol=1e-4)

    def test_restores_loop_after_failure(self):
        loop_forward = self.student.model.forward
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with without_looping(self.student):
                raise RuntimeError("probe")
        self.assertIs(self.student.model.forward, loop_forward)


if __name__ == "__main__":
    unittest.main()
