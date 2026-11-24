"""Predict a dual-readout fusion weight for each example."""

import torch
from torch import nn


MIX_NETWORK_FILENAME = "dual_readout_mix.pt"


class DualReadoutMix(nn.Module):
    def __init__(self, initial_mix):
        super().__init__()
        self.initial_mix = initial_mix
        self.option_features = nn.Sequential(nn.Linear(2, 16), nn.GELU())
        self.log_multiplier = nn.Linear(16, 1)
        nn.init.zeros_(self.log_multiplier.weight)
        nn.init.zeros_(self.log_multiplier.bias)

    def forward(self, numeric, letters):
        scores = torch.stack((numeric, letters), dim=-1).float()
        valid = (numeric != float('-inf')).unsqueeze(-1)
        count = valid.sum(dim=-2, keepdim=True)
        scores = scores.masked_fill(~valid, 0)
        relative_scores = scores - scores.sum(dim=-2, keepdim=True) / count
        # Pool shared option features so changing the option order leaves mix unchanged.
        features = self.option_features(relative_scores).masked_fill(~valid, 0).sum(dim=-2) / count.squeeze(-2)
        return self.initial_mix * (2 * self.log_multiplier(features).tanh()).exp()
