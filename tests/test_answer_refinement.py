import unittest

import torch

from loop_jev.looping import enable_looping, without_looping
from loop_jev.transition import Qwen3LoopForCausalLM
from test_transition_head import config


class AnswerRefinementTests(unittest.TestCase):
    def test_zero_correction_is_standard_pass_and_correction_does_not_move_anchor(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        cfg = config()
        cfg.loop_transition_rank = 0
        cfg.loop_refinement_token_ids = [5, 6, 7, 8]
        cfg.tie_word_embeddings = True
        model = Qwen3LoopForCausalLM(cfg).train()
        enable_looping(model, 1, 2, 3, damping=True, collect_all_readouts=False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ids = torch.tensor([[1, 4, 7, 3], [0, 8, 6, 3]])
        mask = ids.ne(0).long()
        kwargs = dict(attention_mask=mask, logits_to_keep=1)
        with without_looping(model), torch.no_grad():
            direct = model(ids, **kwargs).logits
        initial = model(ids, refine_answer_scores=True, **kwargs)
        torch.testing.assert_close(initial.answer_token_logits, direct)
        torch.testing.assert_close(initial.logits, direct[:, -1, cfg.loop_refinement_token_ids])
        initial.logits[:, 0].sum().backward()
        self.assertGreater(model.loop_answer_refinement.weight.grad.abs().sum().item(), 0)

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.loop_answer_refinement.weight.normal_()
        refined = model(ids, refine_answer_scores=True, **kwargs)
        torch.testing.assert_close(refined.answer_token_logits, direct)
        self.assertFalse(torch.allclose(refined.logits, initial.logits))
        refined.logits[:, 0].sum().backward()
        gradient = model.get_input_embeddings().weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertIsNone(torch.autograd.grad(refined.logits.sum(), refined.answer_token_logits,
                                             allow_unused=True)[0])
