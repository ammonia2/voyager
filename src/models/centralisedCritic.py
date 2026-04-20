from __future__ import annotations
import torch
import torch.nn as nn

from src.utils.obsUtils import GLOBAL_STATE_DIM


class CentralisedCritic(nn.Module):
    """
    MAPPO centralised critic.
    Input:  global state (NUM_AGENTS * OBS_DIM = 441)
    Output: scalar value estimate V(s)
    One shared critic used for advantage computation across all predators.
    """

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
        """
        globalState: (B, GLOBAL_STATE_DIM=441)
        returns:     (B, 1)
        """
        return self.net(globalState)