import copy
import unittest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM
from loop_jev.training.progressive_text import (
    curriculum_stage, retained_steps, encode_trajectories,
    collate_trajectories, trajectory_loss,
)


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class ProgressiveTextTest(unittest.TestCase):
    def test_schedule_and_complete_steps(self):
        stages = [curriculum_stage(i, 192) for i in range(192)]
        self.assertEqual([stages.count(k) for k in range(4)], [48] * 4)
        self.assertEqual(retained_steps(['a', 'b', 'c'], 0), ['a', 'b', 'c'])
        self.assertEqual(retained_steps(['a', 'b', 'c'], 1), ['b', 'c'])
        self.assertEqual(retained_steps(['a', 'b', 'c'], 2), ['c'])
        self.assertEqual(retained_steps(['a'], 2), ['a'])
        self.assertEqual(retained_steps(['a'], 3), [])
        with self.assertRaises(ValueError):
            curriculum_stage(0, 3)

    def test_removed_text_absent_and_causal_labels(self):
        stages = encode_trajectories(CharTokenizer(), [1, 2, 3], 'AAA\nBBB\nCCC', 'B', 200)
        self.assertIn('AAA', ''.join(map(chr, stages[0]['input_ids'][3:])))
        self.assertNotIn('AAA', ''.join(map(chr, stages[1]['input_ids'][3:])))
        self.assertNotIn('BBB', ''.join(map(chr, stages[2]['input_ids'][3:])))
        for item in stages:
            self.assertEqual(item['labels'][:2], [-100, -100])
            self.assertEqual(item['labels'][2], ord('R'))
            self.assertEqual(item['labels'][-1], ord('B'))
            self.assertEqual(len(item['labels']), len(item['input_ids']))
        with self.assertRaises(ValueError):
            encode_trajectories(CharTokenizer(), [1, 2, 3], 'AAA', 'A', 4)

    def test_target_cap_padding_and_reproducibility(self):
        first = encode_trajectories(CharTokenizer(), [1, 2, 3], 'AAA', 'A', 200)[0]
        second = encode_trajectories(CharTokenizer(), [1], '', 'B', 200)[0]
        args = ([first, second], 0, 5, 'cpu')
        packed, targets, owners = collate_trajectories(*args, torch.Generator().manual_seed(42))
        again, targets2, _ = collate_trajectories(*args, torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(packed['logits_to_keep'], again['logits_to_keep']))
        self.assertTrue(torch.equal(targets, targets2))
        self.assertEqual(owners.bincount().tolist(), [5, 5])
        self.assertFalse((packed['logits_to_keep'] & (packed['attention_mask'] == 0)).any())
        self.assertEqual(packed['input_ids'][0].tolist(), first['input_ids'])

    def test_masked_loss_matches_full_and_sequential_backward(self):
        torch.manual_seed(42)
        config = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=8, attention_dropout=0.0,
                            use_cache=False)
        model = Qwen3ForCausalLM(config).train()
        reference = copy.deepcopy(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        examples = [encode_trajectories(CharTokenizer(), [1, 2], 'a', 'A', 100)[0],
                    encode_trajectories(CharTokenizer(), [1], 'bc', 'B', 100)[0]]
        batch, targets, owners = collate_trajectories(examples, 0, 128, 'cpu')
        direct_ids = torch.tensor([[1, 2, 3], [1, 4, 5]])
        F.cross_entropy(model(input_ids=direct_ids, logits_to_keep=1).logits[:, -1],
                        torch.tensor([65, 66])).backward()
        outputs = model(**batch)
        loss = trajectory_loss(outputs.logits, targets, owners, 2)
        loss.backward()
        full = reference(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        full_loss = trajectory_loss(full[batch['logits_to_keep']], targets, owners, 2)
        direct_loss = F.cross_entropy(reference(input_ids=direct_ids, logits_to_keep=1).logits[:, -1],
                                      torch.tensor([65, 66]))
        (full_loss + direct_loss).backward()
        torch.testing.assert_close(loss, full_loss)
        for param, ref in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(param.grad, ref.grad, atol=1e-5, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
