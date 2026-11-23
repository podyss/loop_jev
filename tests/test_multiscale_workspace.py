import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3ForCausalLM

from loop_jev.looping import enable_configured_looping, enable_looping
from loop_jev.multiscale_workspace import exchange_workspace
from test_transition_head import config


class MultiscaleWorkspaceTests(unittest.TestCase):
    def test_exchange_moves_only_the_destination_slots(self):
        previous = torch.zeros(1, 29, 2)
        after_local = previous.clone()
        after_local[:, 3:21] = torch.arange(18).view(1, 18, 1)
        up = exchange_workspace(after_local, previous, 2, upward=True)
        torch.testing.assert_close(up[:, 21:27, 0], torch.arange(1., 18., 3.).view(1, 6))
        torch.testing.assert_close(up[:, :21], after_local[:, :21])
        after_coarse = up.clone()
        after_coarse[:, 21:27] += torch.tensor([2., -1.])
        down = exchange_workspace(after_coarse, up, 2, upward=False)
        torch.testing.assert_close(down[:, 3:21] - after_coarse[:, 3:21],
                                   torch.tensor([2., -1.]).expand(1, 18, 2))
        torch.testing.assert_close(down[:, :3], after_coarse[:, :3])
        torch.testing.assert_close(down[:, 21:], after_coarse[:, 21:])

    def test_round_schedule_and_no_loop(self):
        torch.set_num_threads(1)
        model = Qwen3ForCausalLM(config()).eval()
        ids = torch.tensor([[1] * 29])
        with torch.no_grad():
            plain = model(ids, use_cache=False).logits
            with patch("loop_jev.multiscale_workspace.exchange_workspace", wraps=exchange_workspace) as exchange:
                for mode, directions in (("up", [True]), ("up-down", [True, False])):
                    enable_looping(model, 0, 2, 3, damping=True, collect_all_readouts=False,
                                   multiscale_workspace=(mode, 2))
                    model(ids, use_cache=False)
                    self.assertEqual([call.kwargs["upward"] for call in exchange.call_args_list], directions)
                    exchange.reset_mock()
                enable_configured_looping(model, bypass_looping=True)
                torch.testing.assert_close(model(ids, use_cache=False).logits, plain)
                exchange.assert_not_called()
