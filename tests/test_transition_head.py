import tempfile
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from loop_jev.looping import enable_looping, enable_configured_looping
from loop_jev.transition import Qwen3LoopForCausalLM, load_loop_model


def config():
    return Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=3, num_attention_heads=2,
                       num_key_value_heads=2, head_dim=8,
                       attention_dropout=0.0, use_cache=False,
                       loop_transition_rank=4)


class TransitionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.ids = torch.tensor([[1, 4, 7, 3]])

    def test_identity_and_between_loop_placement(self):
        model = Qwen3LoopForCausalLM(config()).eval()
        baseline = Qwen3ForCausalLM(config()).eval()
        baseline.load_state_dict(model.state_dict(), strict=False)
        enable_looping(model, 1, 2, 3, damping=True)
        enable_looping(baseline, 1, 2, 3, damping=True)
        calls = []
        model.loop_transition.register_forward_hook(lambda *args: calls.append(1))
        torch.testing.assert_close(model(self.ids).logits, baseline(self.ids).logits,
                                   rtol=0, atol=0)
        self.assertEqual(len(calls), 2)
        # K=1 has no transition, even with nonzero learned weights.
        with torch.no_grad():
            model.loop_transition.up.weight.fill_(0.1)
        enable_looping(model, 1, 2, 1, damping=True)
        enable_looping(baseline, 1, 2, 1, damping=True)
        calls.clear()
        torch.testing.assert_close(model(self.ids).logits, baseline(self.ids).logits)
        self.assertEqual(calls, [])

    def test_checkpointed_backward_trains_both_projections(self):
        model = Qwen3LoopForCausalLM(config()).train()
        enable_looping(model, 1, 2, 3, damping=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        head = model.loop_transition
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model(self.ids, labels=self.ids).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(head.up.weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(head.down.weight.grad.abs().sum().item(), 0)
            optimizer.step()

    def test_final_only_readout_preserves_logits_and_gradients(self):
        full = Qwen3LoopForCausalLM(config()).train()
        with torch.no_grad():
            full.loop_transition.up.weight.normal_(std=0.1)
        final = Qwen3LoopForCausalLM(config()).train()
        final.load_state_dict(full.state_dict())
        enable_looping(full, 1, 2, 3, damping=True)
        enable_looping(final, 1, 2, 3, damping=True, collect_all_readouts=False)
        full.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        final.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        left, right = full(self.ids, labels=self.ids), final(self.ids, labels=self.ids)
        torch.testing.assert_close(left.logits, right.logits, rtol=0, atol=0)
        left.loss.backward()
        right.loss.backward()
        self.assertEqual(len(full.model.loop_hidden_states), 3)
        self.assertEqual(len(final.model.loop_hidden_states), 1)
        for p, q in zip(full.parameters(), final.parameters()):
            torch.testing.assert_close(p.grad, q.grad)

    def test_save_reload_and_disabled_looping(self):
        model = Qwen3LoopForCausalLM(config()).eval()
        with torch.no_grad():
            model.loop_transition.up.weight.normal_(std=0.1)
        enable_looping(model, 1, 2, 3, damping=True)
        expected = model(self.ids).logits.detach()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = load_loop_model(directory).eval()
            enable_configured_looping(restored)
            torch.testing.assert_close(restored(self.ids).logits, expected, rtol=0, atol=0)
            disabled = load_loop_model(directory).eval()
            plain = Qwen3ForCausalLM(config()).eval()
            plain.load_state_dict(disabled.state_dict(), strict=False)
            torch.testing.assert_close(disabled(self.ids).logits, plain(self.ids).logits)
            self.assertFalse(torch.allclose(disabled(self.ids).logits, expected))

    def test_bypass_transition_preserves_three_loops_and_weights(self):
        model = Qwen3LoopForCausalLM(config()).eval()
        with torch.no_grad():
            model.loop_transition.up.weight.normal_(std=0.1)
        baseline = Qwen3ForCausalLM(config()).eval()
        baseline.load_state_dict(model.state_dict(), strict=False)
        enable_looping(model, 1, 2, 3, damping=True)
        enable_looping(baseline, 1, 2, 3, damping=True, collect_all_readouts=False)
        original_weights = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = load_loop_model(directory).eval()
            calls = []
            restored.loop_transition.register_forward_hook(lambda *args: calls.append("head"))
            restored.model.layers[1].register_forward_hook(lambda *args: calls.append("block"))
            enable_configured_looping(restored, use_transition=False)
            bypassed = restored(self.ids).logits
            torch.testing.assert_close(bypassed, baseline(self.ids).logits, rtol=0, atol=0)
            self.assertEqual(calls, ["block"] * 3)
            self.assertEqual(restored.config.loop_count, 3)
            self.assertTrue(restored.config.loop_damping)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, original_weights[name], rtol=0, atol=0)
            enable_configured_looping(restored)
            self.assertFalse(torch.allclose(restored(self.ids).logits, bypassed))

    def test_answer_slot_updates_only_last_position(self):
        cfg = config()
        cfg.loop_transition_scope = "answer-slot"
        model = Qwen3LoopForCausalLM(cfg)
        head = model.loop_transition
        inputs = torch.randn(2, 5, cfg.hidden_size, requires_grad=True)
        torch.testing.assert_close(head(inputs), inputs, rtol=0, atol=0)
        with torch.no_grad():
            head.up.weight.normal_(std=0.1)
        seen_shapes = []
        head.down.register_forward_pre_hook(lambda module, args: seen_shapes.append(args[0].shape))
        outputs = head(inputs)
        self.assertEqual(seen_shapes, [torch.Size([2, 1, cfg.hidden_size])])
        torch.testing.assert_close(outputs[:, :-1], inputs[:, :-1], rtol=0, atol=0)
        self.assertFalse(torch.allclose(outputs[:, -1], inputs[:, -1]))
        outputs.sum().backward()
        torch.testing.assert_close(inputs.grad[:, :-1], torch.ones_like(inputs.grad[:, :-1]))
        self.assertGreater(head.up.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.down.weight.grad.abs().sum().item(), 0)
        # Single-token input must also work.
        self.assertEqual(head(inputs[:, -1:]).shape, (2, 1, cfg.hidden_size))

    def test_answer_slot_checkpointed_training_and_reload(self):
        cfg = config()
        cfg.loop_transition_scope = "answer-slot"
        model = Qwen3LoopForCausalLM(cfg).train()
        enable_looping(model, 1, 2, 3, damping=True, collect_all_readouts=False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            # Supervise only the final answer position, as in the real trainer.
            logits = model(self.ids).logits[:, -1, :]
            torch.nn.functional.cross_entropy(logits, torch.tensor([2])).backward()
            self.assertGreater(model.loop_transition.up.weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertGreater(model.loop_transition.down.weight.grad.abs().sum().item(), 0)
        model.eval()
        expected = model(self.ids).logits.detach()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = load_loop_model(directory).eval()
            enable_configured_looping(restored)
            self.assertEqual(restored.loop_transition.scope, "answer-slot")
            torch.testing.assert_close(restored(self.ids).logits, expected, rtol=0, atol=0)
            enable_configured_looping(restored, use_transition=False)
            self.assertFalse(torch.allclose(restored(self.ids).logits, expected))

    def test_loading_base_checkpoint_keeps_identity_initialization(self):
        cfg = config()
        cfg.loop_transition_rank = 0
        base = Qwen3ForCausalLM(cfg)
        with tempfile.TemporaryDirectory() as directory:
            base.save_pretrained(directory)
            cfg.loop_transition_rank = 4
            model = load_loop_model(directory, config=cfg)
            head = model.loop_transition
            self.assertEqual(torch.count_nonzero(head.up.weight).item(), 0)
            self.assertGreater(torch.count_nonzero(head.down.weight).item(), 0)
            self.assertTrue(torch.isfinite(head.norm.weight).all())


if __name__ == "__main__":
    unittest.main()
