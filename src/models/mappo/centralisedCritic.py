from __future__ import annotations
import torch
import torch.nn as nn

from src.utils.obsUtils import GLOBAL_STATE_DIM


class CentralisedCritic(nn.Module):
    def __init__(self, hiddenDim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GLOBAL_STATE_DIM, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, 1),
        )

    def forward(self, globalState: torch.Tensor) -> torch.Tensor:
        return self.net(globalState)
