import tempfile
import unittest

import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from loop_jev.evaluation.short_loop_exit import calibrate
from loop_jev.looping import enable_configured_looping, enable_looping
from loop_jev.training.loop_removal import configure_stage, removal_schedule, trim_workspace, validate_checkpoint


class LoopRemovalTests(unittest.TestCase):
    def test_schedule(self):
        self.assertEqual(removal_schedule("abrupt", 64), ((0, 1, 0),))
        self.assertEqual(removal_schedule("gradual", 64),
                         ((0, 10, 24), (16, 5, 12), (32, 2, 4), (48, 1, 0)))
        self.assertEqual(removal_schedule("gradual", 64, final_rounds=2, keep_slots=True),
                         ((0, 10, 24), (16, 5, 24), (32, 2, 24)))
        with self.assertRaises(ValueError):
            removal_schedule("gradual", 3)

    def test_validates_checkpoint(self):
        cfg = Qwen3Config(loop_schedule="block", loop_start_layer=9, loop_end_layer=18,
                          loop_count=10, loop_damping=True, loop_damping_step=0.1,
                          latent_slots=24, latent_marker_token_id=30)
        validate_checkpoint(cfg, 30)
        with self.assertRaises(ValueError):
            validate_checkpoint(cfg, 29)
        cfg.loop_carry_workspace = True
        with self.assertRaises(ValueError):
            validate_checkpoint(cfg, 30)

    def test_ce_backward_and_native_export(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
                          head_dim=8, attention_dropout=0, use_cache=False)
        model = Qwen3ForCausalLM(cfg).train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        native_forward = model.model.forward
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = dict(input_ids=torch.tensor([[1, 4, *([30] * 24), 8, 3],
                                             [0, 7, *([30] * 24), 8, 3]]))
        batch["attention_mask"] = batch["input_ids"].ne(0).long()
        calls = []
        for index, layer in enumerate(model.model.layers):
            layer.register_forward_pre_hook(lambda *_args, index=index: calls.append(index))
        for _, rounds, slots in removal_schedule("gradual", 4):
            configure_stage(model, rounds, slots, 30, start_layer=1, end_layer=2)
            inputs = trim_workspace(batch, slots, 2)
            self.assertEqual(inputs["input_ids"].tolist(),
                             [[1, 4] + [30] * slots + [8, 3], [0, 7] + [30] * slots + [8, 3]])
            torch.testing.assert_close(inputs["attention_mask"], inputs["input_ids"].ne(0).long())
            optimizer.zero_grad(set_to_none=True)
            calls.clear()
            logits = model(**inputs, logits_to_keep=1).logits[:, -1, :4]
            self.assertEqual(calls, [0] + [1] * rounds + [2])
            F.cross_entropy(logits, torch.tensor([0, 1])).backward()
            gradient = model.model.layers[1].self_attn.q_proj.weight.grad
            self.assertTrue(gradient.isfinite().all())
            self.assertGreater(gradient.abs().sum().item(), 0)
            optimizer.step()
        self.assertEqual(model.model.forward, native_forward)
        with torch.inference_mode(), tempfile.TemporaryDirectory() as root:
            expected = model.eval()(**inputs).logits
            model.save_pretrained(root)
            restored = enable_configured_looping(Qwen3ForCausalLM.from_pretrained(root))
            self.assertEqual((restored.config.loop_count, restored.config.latent_slots), (1, 0))
            self.assertFalse(restored.config.loop_damping)
            self.assertFalse(hasattr(restored.model, "_unlooped_forward"))
            torch.testing.assert_close(restored(**inputs).logits, expected)

    def test_adaptive_exit_only_runs_needed_rounds(self):
        cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
                          head_dim=8, use_cache=False)
        model = Qwen3ForCausalLM(cfg).eval()
        ids = torch.tensor([[1, 4, 7, 3]])
        calls = []
        model.model.layers[1].register_forward_pre_hook(lambda *_: calls.append(1))
        with torch.inference_mode():
            enable_looping(model, 1, 2, 4, damping=True, damping_step=0.25, collect_all_readouts=False)
            full = model(ids).logits
            calls.clear()
            slow = model(ids, block_early_exit=(2, lambda _: False)).logits
            self.assertEqual(len(calls), 4)
            torch.testing.assert_close(full, slow, rtol=0, atol=0)
            calls.clear()
            fast = model(ids, block_early_exit=(2, lambda _: True)).logits
            self.assertEqual(len(calls), 2)
            enable_looping(model, 1, 2, 4, damping=True, damping_step=0.25, collect_all_readouts=False,
                           readout_round=2)
            torch.testing.assert_close(model(ids).logits, fast, rtol=0, atol=0)

    def test_calibration_respects_margin_ties(self):
        records = [(0, 0, [0], [3.0]), (1, 1, [0], [2.0]), (0, 0, [0], [2.0])]
        reference, (_, depth, threshold, correct) = calibrate(records, (3,), 0)
        self.assertEqual((reference, depth, threshold, correct), (3, 3, 3.0, 3))
