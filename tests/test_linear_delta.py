import tempfile
import unittest
import torch
from transformers import Qwen3Config
from loop_jev.transition import Qwen3LoopForCausalLM, load_loop_model
from loop_jev.looping import enable_looping, enable_configured_looping
from loop_jev.training.process_anchors import delta_alignment_loss


class DeltaTests(unittest.TestCase):
    def test_updates_gradient_and_reload(self):
        torch.set_num_threads(1)
        cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=3, num_attention_heads=2,
                         num_key_value_heads=2, head_dim=8, use_cache=False,
                         loop_transition_rank=128, loop_transition_type="linear-delta",
                         loop_transition_scope="answer-slot")
        model = Qwen3LoopForCausalLM(cfg)
        enable_looping(model, 1, 2, 3, damping=True, collect_all_readouts=False, collect_updates=True)
        ids = torch.tensor([[1, 4, 7]])
        model.eval()
        initial = model(ids).logits.detach()
        self.assertEqual(len(model.model.loop_updates), 3)
        enable_configured_looping(model, use_transition=False)
        torch.testing.assert_close(model(ids).logits, initial, rtol=0, atol=0)
        enable_looping(model, 1, 2, 3, damping=True, collect_all_readouts=False, collect_updates=True)
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        calls = []
        handle = model.loop_transition.register_forward_hook(lambda *args: calls.append(1))
        result = model(ids)
        self.assertEqual(len(calls), 3)
        states = [torch.randn(1,16) for _ in range(4)]
        loss = delta_alignment_loss(model.model.loop_updates, states, torch.tensor([True]))
        loss = loss + torch.nn.functional.cross_entropy(result.logits[:, -1], torch.tensor([3]))
        loss.backward()
        self.assertGreater(model.loop_transition.projection.weight.grad.abs().sum().item(), 0)
        handle.remove()
        with torch.no_grad():
            model.loop_transition.projection.weight.normal_(std=0.01)
        model.eval()
        expected = model(ids).logits.detach()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = load_loop_model(directory).eval()
            enable_configured_looping(restored)
            torch.testing.assert_close(restored(ids).logits, expected, rtol=0, atol=0)
            enable_configured_looping(restored, use_transition=False)
            torch.testing.assert_close(restored(ids).logits, initial, rtol=0, atol=0)

    def test_loss_targets_mask_scale_and_detach(self):
        states = [torch.full((2,4), float(k), requires_grad=True) for k in range(4)]
        updates = [torch.ones(2,4,requires_grad=True) for _ in range(3)]
        self.assertEqual(delta_alignment_loss(updates, states, torch.tensor([True,False])).item(), 0)
        zero = [torch.zeros(2,4,requires_grad=True) for _ in range(3)]
        loss = delta_alignment_loss(zero, states, torch.tensor([True,False]))
        self.assertAlmostEqual(loss.item(), 1)
        loss.backward()
        for update in zero:
            self.assertGreater(update.grad[0].abs().sum().item(), 0)
            self.assertEqual(update.grad[1].abs().sum().item(), 0)
        self.assertTrue(all(t.grad is None for t in states))
        empty = delta_alignment_loss(updates, [states[0]] * 4, torch.tensor([False,False]))
        self.assertEqual(empty.item(), 0)
        empty.backward()
