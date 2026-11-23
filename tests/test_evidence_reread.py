import unittest

import torch
from transformers import Qwen3ForCausalLM

from loop_jev.evidence_reread import evidence_bank, reread_evidence
from loop_jev.looping import enable_configured_looping, enable_looping, without_looping
from test_transition_head import config


class EvidenceRereadTests(unittest.TestCase):
    def test_padding_coverage_and_bounded_slot_write(self):
        prefix = torch.tensor([[[3., 0.], [-3., 0.], [1000., 1000.]]])
        valid = torch.tensor([[True, True, False]])
        keys, values = evidence_bank(prefix, valid)
        state = torch.tensor([[[9., 9.], [1., 0.], [7., 7.]]])
        previous = state.clone()
        previous[:, 1] = 0
        updated, first = reread_evidence(state, previous, 2, 1, keys, values, valid, None)
        _, second = reread_evidence(state, previous, 2, 1, keys, values, valid, first)
        self.assertEqual(first[0, 0, 2], 0)
        self.assertGreater(second[0, 0, 1], first[0, 0, 1])
        torch.testing.assert_close(updated[:, (0, 2)], state[:, (0, 2)])
        torch.testing.assert_close(updated[:, 1], torch.tensor([[1.5, 0.]]))

    def test_combined_readout_and_same_weights_ablations(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        cfg = config()
        cfg.latent_slots = 24
        model = Qwen3ForCausalLM(cfg)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ids = torch.randint(1, cfg.vocab_size, (2, 30))
        opts = dict(damping=True, collect_all_readouts=False, multiscale_workspace=("up", 2))
        enable_looping(model, 1, 2, 3, **opts)
        with torch.no_grad():
            up_only = model(ids).logits
            with without_looping(model):
                direct = model(ids).logits
        enable_looping(model, 1, 2, 3, **opts, evidence_reread=(24, 2))
        enable_configured_looping(model)
        logits = model(ids).logits
        self.assertFalse(torch.allclose(logits, up_only))
        logits[:, -1, :].sum().backward()
        self.assertTrue(torch.isfinite(model.get_input_embeddings().weight.grad).all())
        with torch.no_grad():
            torch.testing.assert_close(model(ids, disable_evidence_reread=True).logits, up_only)
            with without_looping(model):
                torch.testing.assert_close(model(ids).logits, direct)
