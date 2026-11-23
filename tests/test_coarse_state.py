import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from loop_jev.evaluation.coarse_state import Backbone, CoarseCorrection, ENDPOINT_MASS, collocate, pack, state_objective
from loop_jev.evaluation.graph_short_loop import lift_nodes, time_quadrature, transport_layer
from loop_jev.looping import enable_looping


class CoarseStateTests(unittest.TestCase):
    def test_zero_initialized_adapter_is_anchor_and_causal(self):
        torch.manual_seed(43)
        model = CoarseCorrection(12, 16)
        before, after = torch.randn(1, 6, 12), torch.randn(1, 6, 12)
        prediction = model(before, after)
        torch.testing.assert_close(prediction, before + 0.45 * (after - before))
        prediction.square().sum().backward()
        self.assertGreater(model.decode.weight.grad.norm().item(), 0)
        torch.nn.init.normal_(model.decode.weight)
        torch.testing.assert_close(model(before, after)[:, :4], model(before[:, :4], after[:, :4]))

    def test_state_objective_ignores_padding(self):
        before = torch.zeros(1, 2, 4)
        target = torch.ones_like(before)
        valid = torch.tensor([[True, False]])
        predicted = target.clone()
        predicted[:, 1] = 10
        self.assertEqual(state_objective(predicted, target, before, valid).item(), 0)

    @torch.inference_mode()
    def test_forward_matches_qwen_and_ten_round_teacher(self):
        cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=19,
                          num_attention_heads=2, num_key_value_heads=2, head_dim=8, use_cache=False)
        cfg._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(cfg).eval()
        labels = [1, 2, 3, 4]
        base = Backbone(model, labels)
        ids, lengths, mask = pack([[1, 8, 3], [4, 9]], 0, 'cpu')
        before, after, ctx = base.inputs(ids, mask)
        native = base.score(after, ctx, lengths)
        enable_looping(model, 9, 18, 10, damping=True, damping_step=0.1, collect_all_readouts=False)
        teacher = base.score(base.teacher(before, after, ctx), ctx, lengths)
        for i, length in enumerate(lengths.tolist()):
            prompt = ids[i:i+1, :length]
            for actual, runner in ((native, model.model._unlooped_forward), (teacher, model.model.forward)):
                hidden = runner(input_ids=prompt, use_cache=False).last_hidden_state
                reference = model.lm_head(hidden)[0, -1, labels]
                torch.testing.assert_close(actual[i], reference, atol=2e-6, rtol=2e-5)

    def test_constant_field_recovers_euler_with_one_parallel_call(self):
        before = torch.tensor([[[1., 2.], [3., 4.]]])
        target = before + 8
        calls = []

        def constant(states):
            calls.append(len(states))
            return target.expand_as(states)

        predicted = time_quadrature(constant, before, target, groups=(4, 5))
        torch.testing.assert_close(predicted, before + (1 - 0.9**10) * (target - before))
        self.assertEqual(calls, [2])

    def test_secant_updates_recover_affine_layer_along_time_direction(self):
        direction = torch.tensor([[[1., 2., 3.]]])
        nodes = torch.arange(4).view(4, 1, 1) * direction
        shifted = nodes + 0.5 * direction

        def layer(inputs):
            return inputs * torch.tensor([2., 3., 4.]) + 1

        torch.testing.assert_close(transport_layer(nodes, layer(nodes), shifted), layer(shifted))
        torch.testing.assert_close(lift_nodes(nodes[:1], nodes[1:2], layer(nodes[:1]), layer(nodes[1:2]),
                                              shifted[1:]), layer(shifted[1:]))

    def test_collocation_evaluates_shared_layer_stack_once(self):
        calls = []

        class Example:
            def stack(self, state, _context, lo, hi):
                calls.append((lo, hi))
                return state + 5

        before = torch.ones(1, 4, 8)
        core = CoarseCorrection(8, 16, midpoint=True)
        output, stage = collocate(Example(), core, before, (None,))
        torch.testing.assert_close(stage, before)
        torch.testing.assert_close(output, before + ENDPOINT_MASS * 5)
        self.assertEqual(calls, [(9, 18)])
