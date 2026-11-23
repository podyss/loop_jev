import unittest

import torch
import torch.nn.functional as F

from loop_jev.evaluation.option_loop_cascade import fast_scores, fit_router, route_score
from loop_jev.evaluation.option_loop_residual import ABLATIONS, CandidateCorrection, ending_parents, feature_mask
from loop_jev.evaluation.option_trie import join_canonical, make_trie, trie_mask
from loop_jev.evaluation.phase_slots import coarsen, consensus_logits, rotate_options, workspace_positions
from loop_jev.evaluation.tail_replay import RecordKV, ReplayKV, tail_positions
from loop_jev.evaluation.two_step_surrogate import TwoStepSurrogate, complete_pair


class TrajectoryCompilationTests(unittest.TestCase):
    def test_candidate_feature_ablation_preserves_columns_and_width(self):
        projection = torch.empty(8, 2)
        features = torch.arange(1, 10).expand(4, -1)
        for name, expected in {'scores-only': [0, 0, 0, 0, 0, 0, 7, 8, 9],
                               'without-candidate': [1, 2, 0, 0, 5, 6, 7, 8, 9],
                               'without-scores': [1, 2, 3, 4, 5, 6, 0, 0, 0]}.items():
            mask = feature_mask(ABLATIONS[name], projection)
            self.assertEqual(features.masked_fill(~mask, 0)[0].tolist(), expected)

    def test_candidate_terminal_states_ignore_padding_and_other_branches(self):
        ids, positions, branches, parents, count, prefix_len = make_trie(
            [10, 11], [[1, 9], [1, 2, 9], [3, 9], [3, 4, 9]], 0)
        ids, positions, branches, label = join_canonical(ids, positions, branches, prefix_len,
                                                         count, [10, 12, 13], 0)
        packed = torch.tensor([ids, positions, branches])
        meta = torch.tensor([count, prefix_len, label, *parents, *([0] * (-count % 16))])
        endpoints = ending_parents(packed, meta)
        self.assertEqual(packed[0, endpoints].tolist(), [1, 2, 3, 4])

    def test_loop_router_uses_option_order_invariant_features(self):
        numeric = torch.tensor([[-1., -2., -3., -4.], [-3., -1., -4., -2.]])
        labels = torch.tensor([[1., -2., 0., -1.], [0., -1., -2., 1.]])
        permutation = [2, 1, 3, 0]
        _, features = fast_scores(numeric, labels)
        _, rotated = fast_scores(numeric[:, permutation], labels[:, permutation])
        torch.testing.assert_close(features, rotated)
        scores = route_score(features, fit_router(features, torch.tensor([1., -1.])))
        self.assertGreater(scores[0].item(), scores[1].item())

    def test_residual_adapter_begins_at_frozen_logits(self):
        head = CandidateCorrection(8)
        features = torch.rand(2, 4, 8)
        scores = torch.randn(2, 4)
        torch.testing.assert_close(head(scores, features), scores)
        F.cross_entropy(head(scores, features), torch.tensor([2, 1])).backward()
        self.assertGreater(head.net[-1].weight.grad.norm().item(), 0)

    def test_trie_matches_separate_causal_branches(self):
        prefix = [10, 11, 12]
        options = [[1, 2], [1, 3, 5], [2, 4], [3, 6]]
        canonical = [10, 11, 20, 21, 22]
        ids, positions, branches, parents, count, prefix_len = make_trie(prefix, options, 0)
        ids, positions, branches, label = join_canonical(ids, positions, branches, prefix_len, count, canonical, 0)
        torch.manual_seed(11)
        hidden = torch.randn(1, 1, len(ids), 8)
        mask = trie_mask(torch.tensor([positions]), torch.tensor([branches]))
        joint = F.scaled_dot_product_attention(hidden, hidden, hidden, attn_mask=mask)
        for i, expected in enumerate([*[prefix + o for o in options], canonical]):
            indices = [j for j, branch in enumerate(branches) if branch & (1 << i)]
            self.assertEqual([ids[j] for j in indices], expected)
            contiguous = hidden[:, :, indices]
            out = F.scaled_dot_product_attention(contiguous, contiguous, contiguous, is_causal=True)
            torch.testing.assert_close(joint[:, :, indices], out)
        self.assertEqual(ids[label], canonical[-1])
        self.assertEqual(parents[0], len(prefix) - 1)

    def test_option_alignment_and_workspace(self):
        row = {'options': ['1', '2', '3', '4'], 'gold_index': 3}
        other, mapping = rotate_options(row, 2)
        self.assertEqual(other['gold_index'], 1)
        index = torch.tensor([mapping])
        original = torch.tensor([[0., 3., 1., -1.]])
        canonical, rotated = consensus_logits(original, original.gather(1, index), index)
        torch.testing.assert_close(canonical.softmax(-1), original.softmax(-1))
        torch.testing.assert_close(rotated, canonical.gather(1, index))
        positions = workspace_positions(torch.tensor([32, 50]), 3)
        self.assertEqual(positions[:, 0].tolist(), [5, 23])
        self.assertEqual(positions[:, -1].tolist(), [28, 46])
        torch.testing.assert_close(coarsen(torch.arange(24.).view(1, 24, 1)).view(-1), torch.arange(1., 24., 3.))

    def test_kv_replay_preserves_prefix_and_refreshes_suffix(self):
        positions = tail_positions(torch.tensor([5]), 8, 2)
        now, previous = RecordKV(), RecordKV()
        state = torch.arange(8.).view(1, 1, 8, 1)
        now.update(state, state * 2, 9)
        previous.update(state - 1, (state - 1) * 2, 9)
        tail = torch.full((2, 1, 2, 1), 10.)
        replay = ReplayKV(now, positions, previous, torch.tensor([1.5, 5.5]))
        key, value = replay.update(tail, tail * 2, 9)
        torch.testing.assert_close(key[:, 0, :3, 0], torch.tensor([[1.5, 2.5, 3.5], [5.5, 6.5, 7.5]]))
        torch.testing.assert_close(key[:, 0, 3:5, 0], torch.full((2, 2), 10.))
        torch.testing.assert_close(value, 2 * key)

    def test_surrogate_initialization_keeps_constant_map_and_gradients(self):
        h0 = torch.randn(2, 5, 16)
        raw = torch.randn_like(h0)
        h1, h2 = torch.lerp(h0, raw, .1), torch.lerp(h0, raw, .19)
        after = torch.lerp(h0, raw, 1 - .9**10)
        torch.testing.assert_close(complete_pair(h0, h1, h2)[-1], after)
        core = TwoStepSurrogate(16, 24)
        prediction = core(h0, h1, h2, None)
        torch.testing.assert_close(prediction, after)
        prediction.square().mean().backward()
        self.assertGreater(core.correction.weight.grad.norm().item(), 0.)


if __name__ == '__main__':
    unittest.main()
