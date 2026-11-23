import unittest

import torch
from transformers import Qwen3ForCausalLM

from loop_jev.looping import enable_looping
from loop_jev.training.workspace_dependency import dependency
from test_transition_head import config


class WorkspaceDependencyTests(unittest.TestCase):
    def test_extracts_only_checked_nontrivial_edges(self):
        self.assertEqual(dependency("<<48/2=24>> ... <<48+24=72>>"), ("48 + x", "24", "72"))
        self.assertEqual(dependency("<<12/60=0.2>> ... <<0.2*50=10>>"), ("x * 50", "0.2", "10"))
        self.assertIsNone(dependency("<<48/2=23>> ... <<48+23=71>>"))
        self.assertIsNone(dependency("<<48/2=24>> ... <<24*0=0>>"))
        self.assertIsNone(dependency("<<48/2=24>> ... <<24+24=48>>"))

    def test_reader_backpropagates_to_writer_without_changing_answer_path(self):
        torch.set_num_threads(1)
        model = Qwen3ForCausalLM(config()).train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        enable_looping(model, 1, 2, 3, damping=True, collect_all_readouts=False, workspace_slice=slice(-3, -1))
        ids = torch.tensor([[1, 4, 7, 3], [2, 4, 7, 3]])
        reader = torch.tensor([[5, 6], [5, 0]])
        mask = torch.tensor([[1, 1], [1, 0]])
        with torch.no_grad():
            expected = model(ids).logits
        actual = model(ids, dependency_input=(reader, mask, True)).logits
        torch.testing.assert_close(actual, expected)
        _, real, blank = model.model.loop_dependency_states
        first_workspace = model.model.loop_workspace_states[0]
        first_workspace.retain_grad()
        self.assertFalse(torch.allclose(real, blank))
        self.assertFalse(blank.requires_grad)
        real.square().mean().backward()
        self.assertGreater(first_workspace.grad.abs().sum().item(), 0)
        early_gradient = model.model.layers[0].self_attn.q_proj.weight.grad
        self.assertGreater(early_gradient.abs().sum().item(), 0)
