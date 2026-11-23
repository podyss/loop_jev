import unittest
import torch

from loop_jev.evaluation.packed_linear import pack_groups
from loop_jev.evaluation.time_fold import FoldLinear, fold_targets
from loop_jev.evaluation.time_nodes import TimeNodes


def _check_fold_adds_per_depth_contributions():
    before = torch.randn(2, 4, 8)
    targets = fold_targets(lambda state, lo, hi: state + hi - lo, before)
    for state, total in zip(targets, (3, 6, 9), strict=True):
        torch.testing.assert_close(state, before + total)


def _check_low_rank_merge_has_no_residual_adapter():
    layer = FoldLinear(torch.nn.Linear(8, 12, bias=False), 4)
    torch.nn.init.normal_(layer.up)
    state = torch.randn(2, 4, 8)
    reference = layer(state)
    merged = layer.merge()
    assert isinstance(merged, torch.nn.Linear)
    torch.testing.assert_close(merged(state), reference)


def _check_node_correction_makes_one_batched_block_call():
    before = torch.randn(2, 4, 8)
    core = TimeNodes(8, 16, 15)
    calls = []

    class ConstantBlock:
        def stack(self, states, context, lo, hi):
            calls.append((lo, hi, states.shape, context[-1].shape))
            return torch.cat((before, before)) + 5

    mask = torch.ones(2, 1, 4, 4, dtype=torch.bool).tril()
    result, _ = core.advance(ConstantBlock(), before, before + 1, before + 5, (None, None, mask))
    torch.testing.assert_close(result, before + (1 - .9**10) * 5)
    assert calls == [(15, 18, (4, 4, 8), (4, 1, 4, 4))]


def _check_node_predictor_is_causal_and_trainable():
    torch.manual_seed(43)
    core = TimeNodes(8, 24, 15)
    torch.nn.init.normal_(core.outputs.weight)
    inputs = tuple(torch.randn(2, 4, 8) for _ in range(3))
    outputs = core(*inputs, None)
    for long, short in zip(outputs, core(*(x[:, :2] for x in inputs), None), strict=True):
        torch.testing.assert_close(long[:, :2], short)
    sum(t.square().mean() for t in outputs).backward()
    assert all(p.grad is not None and p.grad.norm() > 0 for p in core.parameters())


def _check_packed_codes_roundtrip(bits):
    bias = 1 << (bits - 1)
    codes = torch.arange(64 * 256).reshape(64, 256) % (2 * bias - 1) - (bias - 1)
    codes[:, ::128] = bias - 1
    weight = codes.float() * torch.tensor([.125, .25]).repeat_interleave(128)
    packed, scales = pack_groups(weight, bits, 128)
    factor = 32 // bits
    decoded = torch.stack([(packed.T >> (bits * i)) & (2 * bias - 1) for i in range(factor)], -1)
    reconstructed = (decoded.reshape_as(codes) - bias) * scales.T.repeat_interleave(128, dim=-1)
    torch.testing.assert_close(reconstructed, weight)


class StepCompressionTests(unittest.TestCase):
    def test_fold_adds_per_depth_contributions(self):
        _check_fold_adds_per_depth_contributions()

    def test_low_rank_merge_has_no_residual_adapter(self):
        _check_low_rank_merge_has_no_residual_adapter()

    def test_node_correction_makes_one_batched_block_call(self):
        _check_node_correction_makes_one_batched_block_call()

    def test_node_predictor_is_causal_and_trainable(self):
        _check_node_predictor_is_causal_and_trainable()

    def test_packed_codes_roundtrip(self):
        for bits in (4, 8):
            with self.subTest(bits=bits):
                _check_packed_codes_roundtrip(bits)
