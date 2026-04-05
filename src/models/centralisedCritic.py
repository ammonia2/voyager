from __future__ import annotations
import torch
import torch.nn as nn

from src.utils.obsUtils import OBS_DIM, NUM_AGENTS, ACTION_ONEHOT_DIM

STATE_DIM        = NUM_AGENTS * OBS_DIM                         # 225
CRITIC_INPUT_DIM = STATE_DIM + NUM_AGENTS * ACTION_ONEHOT_DIM  # 225 + 24 = 249


class CentralizedQNetwork(nn.Module):
    """
    Per-agent centralized Q-network (paper eq. 9-10).
    Input:  global state (225) + all agents' action one-hots (24) = 249
    Output: scalar Q-value (B, 1)
    """

    def __init__(self, inputDim: int = CRITIC_INPUT_DIM, hiddenDim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, 1),
        )

    def forward(self, state: torch.Tensor, actionsOnehot: torch.Tensor) -> torch.Tensor:
        """
        state:         (B, 225)
        actionsOnehot: (B, 24)
        returns:       (B, 1)
        """
        return self.net(torch.cat([state, actionsOnehot], dim=-1))