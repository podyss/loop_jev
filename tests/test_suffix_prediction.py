import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from loop_jev.training.suffix_prediction import pack_suffixes, suffix_texts, suffix_prediction_loss


class SuffixTest(unittest.TestCase):
    def test_curriculum_preserves_whole_steps(self):
        self.assertEqual(suffix_texts(['a', 'b', 'c']), ['a\nb\nc', 'b\nc', 'c'])
        self.assertEqual(suffix_texts(['a']), ['a'] * 3)
        self.assertEqual(suffix_texts([]), [''] * 3)

    def test_shift_and_mask(self):
        ids, mask, labels = pack_suffixes([[4, 5, 6], []], 2, 0, 'cpu')
        self.assertEqual(ids.tolist(), [[0, 0, 4, 5], [0, 0, 0, 0]])
        self.assertEqual(labels.tolist(), [[-100, 4, 5, 6], [-100] * 4])
        self.assertEqual(mask.tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])

    def test_frozen_decoder_backprop_and_empty_batch(self):
        torch.manual_seed(42)
        decoder = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.0)).eval()
        decoder.requires_grad_(False)
        states = [torch.randn(2, 16, requires_grad=True) for _ in range(3)]
        loss = suffix_prediction_loss(decoder, states, [[4, 5, 6], []], 0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for state in states:
            self.assertGreater(state.grad[0].abs().sum().item(), 0)
            self.assertEqual(state.grad[1].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))
        self.assertEqual(len(decoder.get_input_embeddings()._forward_hooks), 0)
        states = [torch.randn(2, 16, requires_grad=True)]
        loss = suffix_prediction_loss(decoder, states, [[], []], 0)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(states[0].grad.abs().sum().item(), 0)

    def test_student_checkpointed_single_pass_receives_suffix_gradients(self):
        from loop_jev.looping import enable_looping
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, attention_dropout=0.0)
        student = Qwen3ForCausalLM(config).train()
        decoder = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        enable_looping(student, 0, 2, 1, damping=True, collect_all_readouts=False)
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        captures = {}
        handles = []
        for k, layer in enumerate(student.model.layers):
            def capture(_module, _args, output, index=k):
                captures[index] = output[:, -1].clone()
            handles.append(layer.register_forward_hook(capture))
        outputs = student(input_ids=torch.tensor([[2, 3, 4], [3, 5, 6]]),
                          use_cache=False, logits_to_keep=1)
        loss = suffix_prediction_loss(decoder, [captures[k] for k in range(3)],
                                      [[7, 8], [9, 10]], 0)
        loss.backward()
        for layer in student.model.layers:
            self.assertGreater(layer.self_attn.v_proj.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(outputs.logits).all())
        for handle in handles:
            handle.remove()


if __name__ == '__main__':
    unittest.main()
