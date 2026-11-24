import unittest

import torch

from loop_jev.evaluation.block_shortcut import (
    DIRECT, FIRST, SECOND, IntervalLinear, advance, advance_rounds, choose_train_rows, predict_direct,
    rollout_states, teacher_states, two_hop_predictor,
)


class ConstantBlock:
    def __init__(self, before):
        self.before = before
        self.calls = []

    def prefix(self, ids):
        return self.before, None

    def stack(self, state, context, lo, hi):
        self.calls.append((lo, hi))
        return torch.full_like(state, 5)

    def score(self, state, context, lengths):
        return state[:, 0, :]


class BlockShortcutTests(unittest.TestCase):
    def test_teacher_hops_and_direct_geometric_mass(self):
        before = torch.ones(2, 4, 8)
        base = ConstantBlock(before)
        middle, end = teacher_states(base, before, None)
        self.assertEqual(base.calls, [(9, 18)] * 10)
        first = advance(base, before, None, FIRST)
        composed = advance(base, first, None, SECOND)
        direct = advance(base, before, None, DIRECT)
        torch.testing.assert_close(first, middle)
        torch.testing.assert_close(composed, end)
        torch.testing.assert_close(direct, end)

    def test_predict_direct_runs_the_shared_block_once(self):
        base = ConstantBlock(torch.randn(2, 4, 8))
        predict_direct(base, None, None)
        self.assertEqual(base.calls, [(9, 18)])

    def test_consistency_uses_adjacent_teacher_states_and_an_identity_endpoint(self):
        before = torch.ones(2, 4, 8)
        base = ConstantBlock(before)
        trajectory = teacher_states(base, before, None, rounds=tuple(range(11)))
        self.assertEqual(len(trajectory), 11)
        self.assertIs(trajectory[0], before)
        torch.testing.assert_close(advance_rounds(base, trajectory[9], None, 1), trajectory[10])
        base.calls.clear()
        self.assertIs(advance_rounds(base, trajectory[10], None, 0), trajectory[10])
        self.assertEqual(base.calls, [])

    def test_two_hop_predictor_applies_both_merged_intervals_in_order(self):
        selected = []

        class Core:
            def merged(self, interval):
                return interval

            def replace(self, interval):
                selected.append(interval)

        base = ConstantBlock(torch.randn(2, 4, 8))
        two_hop_predictor(Core())(base, None, None)
        self.assertEqual(selected, [FIRST, SECOND])
        self.assertEqual(base.calls, [(9, 18)] * 2)

    def test_rollout_terminal_loss_differentiates_through_first_hop(self):
        first_weight = torch.tensor(2.0, requires_grad=True)
        second_weight = torch.tensor(3.0, requires_grad=True)

        class Core:
            def hop(self, state, context, interval):
                return state * {FIRST: first_weight, SECOND: second_weight}[interval]

        middle, end = rollout_states(Core(), torch.ones(()), None)
        self.assertEqual(middle.item(), 2)
        self.assertEqual(end.item(), 6)
        end.backward()
        self.assertEqual(first_weight.grad.item(), 3)
        self.assertEqual(second_weight.grad.item(), 2)

    def test_disjoint_split_excludes_all_original_teacher_rows(self):
        def ids(rows):
            return {row['id'] for row in rows}

        pool = [{'id': i} for i in range(3000)]
        seen = ids(choose_train_rows(pool, 1024, 'original'))
        selected = ids(choose_train_rows(pool, 1024, 'disjoint'))
        self.assertEqual(len(selected), 1024)
        self.assertFalse(seen & selected)
        self.assertEqual(selected, ids(choose_train_rows(pool, 1024, 'disjoint')))

    def test_merge_uses_only_selected_interval(self):
        torch.manual_seed(43)
        selector = [DIRECT]
        layer = IntervalLinear(torch.nn.Linear(8, 12).requires_grad_(False), 4, selector)
        torch.nn.init.normal_(layer.up)
        torch.nn.init.normal_(layer.gain)
        state = torch.randn(2, 4, 8)
        direct = layer.merged(DIRECT)
        self.assertIsInstance(direct, torch.nn.Linear)
        torch.testing.assert_close(direct(state), layer(state))
        selector[0] = SECOND
        self.assertFalse(torch.allclose(direct(state), layer(state)))
        torch.testing.assert_close(layer.merged(SECOND)(state), layer(state))

    def test_two_forwards_keep_interval_specific_gradients(self):
        selector = [FIRST]
        layer = IntervalLinear(torch.nn.Linear(8, 12).requires_grad_(False), 4, selector)
        torch.nn.init.normal_(layer.up)
        first = layer(torch.randn(2, 4, 8))
        selector[0] = SECOND
        second = layer(first[..., :8])
        (first.square().mean() + second.square().mean()).backward()
        self.assertEqual(layer.gain.grad[DIRECT].norm().item(), 0)
        self.assertGreater(layer.gain.grad[FIRST].norm().item(), 0)
        self.assertGreater(layer.gain.grad[SECOND].norm().item(), 0)
