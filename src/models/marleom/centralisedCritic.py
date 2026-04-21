from __future__ import annotations
import torch
import torch.nn as nn

from src.utils.obsUtils import GLOBAL_STATE_DIM, N_MOVE, N_TURN, N_ATTACK, PREDATOR_INDICES

PREDATOR_ACTION_DIM = N_MOVE + 1 + N_ATTACK
PREY_ACTION_DIM = N_MOVE + N_TURN + N_ATTACK
JOINT_ACTION_DIM = len(PREDATOR_INDICES) * PREDATOR_ACTION_DIM + PREY_ACTION_DIM


class CentralisedCritic(nn.Module):
    """Centralised Q network for MARLeOM: Q(s, a)."""

    def __init__(self, hiddenDim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GLOBAL_STATE_DIM + JOINT_ACTION_DIM, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, 1),
        )

    def forward(self, globalState: torch.Tensor, actionsOh: torch.Tensor) -> torch.Tensor:
        x = torch.cat([globalState, actionsOh], dim=-1)
        return self.net(x)
