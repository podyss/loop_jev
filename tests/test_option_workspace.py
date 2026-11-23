import unittest

import torch
from torch.nn.functional import cross_entropy

from loop_jev.looping import enable_configured_looping, enable_looping
from loop_jev.option_workspace import SLOT_COUNT, pad_option_spans, seed_option_slots
from loop_jev.transition import Qwen3LoopForCausalLM, load_loop_model
from test_transition_head import config


class OptionWorkspaceTests(unittest.TestCase):
    def test_seeds_corresponding_spans_after_left_padding(self):
        hidden = torch.arange(32.0).view(1, 32, 1)
        spans = pad_option_spans([[(0, 2), (2, 3), (3, 4), (4, 5)]], [2])
        seeded = seed_option_slots(hidden, spans, torch.arange(6.0).view(6, 1), suffix=1)
        torch.testing.assert_close(seeded[:, :7], hidden[:, :7])
        torch.testing.assert_close(seeded[:, 7:-1:6, 0], torch.tensor([[2.5, 4.0, 5.0, 6.0]]))
        torch.testing.assert_close(seeded[:, -1], hidden[:, -1])

    def test_four_way_ce_and_checkpointed_readout(self):
        from tempfile import TemporaryDirectory

        torch.set_num_threads(1)
        cfg = config()
        cfg.option_workspace = True
        cfg.option_workspace_suffix = 1
        model = Qwen3LoopForCausalLM(cfg)
        enable_looping(model, 1, 2, 3, damping=True, collect_all_readouts=False)
        ids = torch.tensor([[1, 2, 3, 4] + [5] * SLOT_COUNT + [6]])
        spans = torch.tensor([[[i, i + 1] for i in range(4)]])
        output = model(ids, option_spans=spans, option_vocab_logits=True)
        self.assertEqual(output.logits.shape, (1, 4))
        self.assertEqual(output.answer_token_logits.shape, (1, 1, cfg.vocab_size))
        cross_entropy(output.logits, torch.tensor([1])).backward()
        for parameter in (model.option_slot_offsets.weight, model.option_scorer.weight,
                          model.model.layers[1].self_attn.q_proj.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = load_loop_model(directory).eval()
            enable_configured_looping(restored)
            with torch.no_grad():
                scored = restored(ids, option_spans=spans)
                self.assertIsNone(scored.answer_token_logits)
                torch.testing.assert_close(scored.logits, output.logits)
                enable_configured_looping(restored, bypass_looping=True)
                off = restored(ids, option_spans=spans).logits
                enable_looping(restored, 1, 2, 1, damping=False, collect_all_readouts=False)
                torch.testing.assert_close(off, restored(ids, option_spans=spans).logits)
