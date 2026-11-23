import unittest

import torch
from transformers import Qwen3ForCausalLM

from loop_jev.looping import enable_looping
from test_transition_head import config


class StagedHandoffTests(unittest.TestCase):
    def test_question_is_transmitted_only_by_slots(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        cfg = config()
        cfg._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(cfg).eval()
        enable_looping(model, 1, 2, 3, damping=True, collect_all_readouts=False, staged_handoff=(2, 1))
        # Different left padding and different option text; all slots line up.
        ids = torch.tensor([[1, 2, 3, 4, 5, 5, 6], [0, 7, 8, 9, 5, 5, 6]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1]])
        offsets = torch.tensor([3, 3])
        with torch.no_grad():
            def run(tokens, padding, swap=False):
                return model(tokens, attention_mask=padding, handoff_option_start=offsets,
                             swap_handoff=swap, logits_to_keep=1).logits

            native = run(ids, mask)
            swapped = run(ids, mask, swap=True)
            changed = torch.cat((ids[:, :3].flip(0), ids[:, 3:]), dim=1)
            new_mask = torch.cat((mask[:, :3].flip(0), mask[:, 3:]), dim=1)
            torch.testing.assert_close(swapped, run(changed, new_mask))
            self.assertGreater((native - swapped).abs().max().item(), 1e-5)

        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model(ids, attention_mask=mask, handoff_option_start=offsets, logits_to_keep=1).logits.sum().backward()
        gradient = model.get_input_embeddings().weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[1].abs().sum().item(), 0)
