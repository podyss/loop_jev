import tempfile
import unittest

import torch
from transformers import Qwen3ForCausalLM

from loop_jev.looping import enable_configured_looping, enable_looping, without_looping
from loop_jev.transition import load_loop_model
from test_transition_head import config


class LayerwiseLoopTests(unittest.TestCase):
    def test_gated_load_from_base_and_checkpointed_gradients(self):
        torch.set_num_threads(1)
        base = Qwen3ForCausalLM(config()).eval()
        ids = torch.tensor([[1, 4, 7, 3]])
        cfg = config()
        cfg.loop_transition_rank = 0
        cfg.loop_schedule, cfg.loop_count = "layerwise-gated", 3
        cfg.loop_start_layer, cfg.loop_end_layer = 0, 2
        with tempfile.TemporaryDirectory() as directory:
            base.save_pretrained(directory)
            model = load_loop_model(directory, config=cfg)
            with torch.no_grad():
                torch.testing.assert_close(base(ids).logits, model(ids).logits, atol=0, rtol=0)
                for layer in model.model.layers[:2]:
                    attention, mlp = layer.loop_gates(1)
                    torch.testing.assert_close(attention, torch.full_like(attention, 0.3))
                    torch.testing.assert_close(mlp, attention)
            enable_configured_looping(model)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            model(ids, labels=ids).loss.backward()
            for layer in model.model.layers[:2]:
                self.assertTrue(layer.loop_gates.raw.grad.abs().sum(dim=-1).gt(0).all())
            with torch.no_grad():
                model.model.layers[0].loop_gates.raw[1, 1].add_(0.1)
                expected = model.eval()(ids).logits
            target = directory + "/updated"
            model.save_pretrained(target)
            restored = enable_configured_looping(load_loop_model(target))
            with torch.no_grad():
                torch.testing.assert_close(expected, restored(ids).logits, atol=0, rtol=0)

    def test_schedule_and_checkpointed_backward(self):
        torch.set_num_threads(1)
        model = Qwen3ForCausalLM(config()).train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        enable_looping(model, 0, 2, 3, collect_all_readouts=False, schedule="layerwise")
        calls = []
        for i, layer in enumerate(model.model.layers):
            layer.register_forward_pre_hook(lambda *args, index=i: calls.append(index))
        ids = torch.tensor([[1, 4, 7, 3]])
        loss = model(ids, labels=ids).loss
        self.assertEqual(calls, [0, 0, 0, 1, 1, 1, 2])
        loss.backward()
        for layer in model.model.layers[:2]:
            gradient = layer.self_attn.q_proj.weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_checkpoint_restores_schedule_and_single_pass_stays_standard(self):
        torch.set_num_threads(1)
        model = Qwen3ForCausalLM(config()).eval()
        ids = torch.tensor([[1, 4, 7, 3]])
        with torch.no_grad():
            plain = model(ids).logits
            enable_looping(model, 0, 2, 3, collect_all_readouts=False, schedule="layerwise")
            looped = model(ids).logits
            self.assertFalse(torch.allclose(plain, looped))
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                restored = enable_configured_looping(Qwen3ForCausalLM.from_pretrained(directory))
                self.assertEqual(restored.config.loop_schedule, "layerwise")
                torch.testing.assert_close(restored(ids).logits, looped)
                with without_looping(restored):
                    torch.testing.assert_close(restored(ids).logits, plain)
                enable_configured_looping(restored, readout_round=1)
                torch.testing.assert_close(restored(ids).logits, plain)
