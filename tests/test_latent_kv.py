import copy
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from loop_jev.evaluation.zero_shot import render_prompt
from loop_jev.looping import enable_looping, without_looping
from loop_jev.training.latent_kv import capture_workspace_kv, compressed_kv, kv_alignment_loss


class ToyTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return [1, 2, 3]

    def encode(self, _text, **_kwargs):
        return [4, 5]


class LatentKVTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        config = Qwen3Config(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=4, num_attention_heads=2,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.0,
            use_cache=False,
        )
        self.model = Qwen3ForCausalLM(config).train()
        enable_looping(self.model, 1, 3, 3, damping=True, collect_all_readouts=False)
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    def test_workspace_positions_shared_between_training_and_evaluation(self):
        ids = render_prompt(ToyTokenizer(), {"question": "x", "options": ["y"]}, 3, 9)
        self.assertEqual(ids, [1, 2, 3, 9, 9, 9, 4, 5])
        captured, hooks = capture_workspace_kv(self.model, 2, 3, 2)
        try:
            logits = self.model(torch.tensor([ids]), logits_to_keep=1).logits
            self.assertEqual(captured["key"].shape, (1, 3, 2, 8))
            self.assertEqual(captured["value"].shape, (1, 3, 2, 8))
            teacher_key = torch.randn_like(captured["key"])
            teacher_value = torch.randn_like(captured["value"])
            loss = logits.square().mean() + kv_alignment_loss(
                captured["key"], captured["value"], teacher_key, teacher_value,
            )
            loss.backward()
            self.assertGreater(self.model.model.layers[2].self_attn.k_proj.weight.grad.abs().sum(), 0)
            captured.clear()
            with without_looping(self.model), torch.no_grad():
                direct = self.model(torch.tensor([ids]), logits_to_keep=1).logits
            self.assertEqual(direct.shape, logits.shape)
        finally:
            for hook in hooks:
                hook.remove()

    def test_pool_repeats_short_traces_and_preserves_heads(self):
        key = torch.arange(2 * 2 * 8).reshape(2, 2, 8)
        value = copy.deepcopy(key)
        pooled_key, pooled_value = compressed_kv(key, value, 4)
        self.assertEqual(pooled_key.shape, (4, 2, 8))
        torch.testing.assert_close(pooled_key, pooled_value)
        self.assertTrue(torch.isfinite(pooled_key).all())


if __name__ == "__main__":
    unittest.main()
