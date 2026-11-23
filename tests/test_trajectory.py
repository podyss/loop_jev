import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from loop_jev.evaluation.short_loop import TEST_SIZE, parse_experiment, validate_test_rows
from loop_jev.evaluation.trajectory_probe import project
from loop_jev.looping import enable_looping
from loop_jev.trajectory import extrapolate, parallel_finish, segmented_extrapolate


class TrajectoryTests(unittest.TestCase):
    def test_rotation_recovers_information_unavailable_to_linear_control(self):
        path = tuple(torch.tensor([[value]], dtype=torch.float32)
                     for value in ([0, 0], [1, 0], [1, 1], [0, 1]))
        target = torch.zeros_like(path[-1])
        linear = extrapolate(path, None, steps=1, method="linear")
        dynamic = extrapolate(path, None, steps=1, method="dmd")
        self.assertLess((dynamic - target).norm().item(), (linear - target).norm().item())

    def test_padding_does_not_affect_fit(self):
        generator = torch.Generator().manual_seed(73)
        path = tuple(torch.randn(2, 4, 8, generator=generator) for _ in range(4))
        mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        expected = extrapolate(tuple(state[:, 2:] for state in path), None, steps=2, method="dmd")
        actual = extrapolate(path, mask, steps=2, method="dmd")
        torch.testing.assert_close(actual[:, 2:], expected)
        split = segmented_extrapolate(path, mask, steps=2, workspace_width=2)
        torch.testing.assert_close(split[:, 2:], expected)
        still = extrapolate((path[0],) * 4, mask, steps=7, method="dmd")
        torch.testing.assert_close(still, path[0], rtol=0, atol=0)

    def test_final_transform_uses_only_three_rounds_and_one_readout(self):
        cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
                          head_dim=8, use_cache=False)
        model = Qwen3ForCausalLM(cfg).eval()
        enable_looping(model, 1, 2, 3, damping=True, damping_step=0.2, collect_all_readouts=False)
        ids = torch.tensor([[0, 4, 7, 3]])
        mask = ids.ne(0).long()
        calls = []
        for index, layer in enumerate(model.model.layers):
            layer.register_forward_pre_hook(lambda *_args, index=index: calls.append(index))

        def identity(states, supplied_mask, _evaluate):
            self.assertEqual(len(states), 4)
            self.assertIs(supplied_mask, mask)
            return states[-1]

        with torch.inference_mode():
            reference = model(ids, attention_mask=mask).logits
            calls.clear()
            actual = model(ids, attention_mask=mask, block_final_transform=identity).logits
            self.assertEqual(calls, [0, 1, 1, 1, 2])
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model(ids, attention_mask=mask, block_final_transform=identity)

    def test_modes(self):
        self.assertEqual(parse_experiment("3:0.2:dmd:2"), ((3, 0.2), "dmd", 2))
        self.assertEqual(parse_experiment("2:0.25"), ((2, 0.25), None, None))
        self.assertEqual(parse_experiment("native"), ((1, None), None, None))

    def test_evaluation_requires_all_test_questions(self):
        rows = [dict(id=f"gsm-mc:test:{i}", split="test", options=list("ABCD"), gold_index=0)
                for i in range(TEST_SIZE)]
        validate_test_rows(rows)
        with self.assertRaises(ValueError):
            validate_test_rows(rows[:-1])
        with self.assertRaises(ValueError):
            validate_test_rows([*rows[:-1], rows[0]])
        with self.assertRaises(ValueError):
            validate_test_rows([*rows[:-1], dict(rows[-1], split="train")])

    def test_oracle_recovers_in_span_update_but_not_orthogonal_components(self):
        path = tuple(torch.tensor([[value]], dtype=torch.float32)
                     for value in ([0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]))
        target = torch.tensor([[[4.0, 3.0, 5.0]]])
        torch.testing.assert_close(project(path, target), torch.tensor([[[2.0, 3.0, 0.0]]]),
                                   rtol=2e-4, atol=1e-6)

    def test_parallel_finish_batches_nodes_and_reproduces_constant_velocity(self):
        path = tuple(torch.full((1, 1, 2), float(i)) for i in range(3))
        calls = []

        def evaluate(states):
            calls.append(states[:, 0, 0].tolist())
            return states + 10

        result = parallel_finish(path, evaluate, rounds=8, step=0.1, nodes=4, sweeps=2)
        self.assertEqual(calls, [[2.5, 4.5, 6.5, 8.5], [2.5, 4.5, 6.5, 8.5]])
        torch.testing.assert_close(result, torch.full_like(result, 10))
