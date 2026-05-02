"""
masac/centralisedCritic.py
==========================
Twin soft Q-network (centralised critic) for MASAC.

Architecture
------------
SAC uses TWO Q-networks (Q1, Q2) and takes the minimum to reduce
overestimation bias (Haarnoja et al. 2018).

For MULTI-AGENT SAC we use a CENTRALISED critic that observes the global
state (all agents' observations concatenated) AND the joint predator action.
This is analogous to the CTDE (Centralised Training / Decentralised
Execution) paradigm used in MAPPO.

Input
-----
  globalState  : (B, GLOBAL_STATE_DIM=441)   — all 3 agents' flat obs concat'd
  jointAction  : (B, JOINT_ACTION_DIM)        — both predators' continuous actions

  Predator action encoding per agent (for Q-input):
    move_oh (3) + turn_cont (1) + attack_oh (2) = 6-dim
  Joint action = 2 predators × 6 = 12-dim

  GLOBAL_STATE_DIM = 3 × OBS_DIM = 3 × 147 = 441

Output
------
  Q1_value : (B, 1)
  Q2_value : (B, 1)

Usage
-----
  • During update: min(Q1, Q2) for the actor gradient target
  • Target networks: a slow-moving EMA copy of the twin Q-net
"""
from __future__ import annotations
import torch
import torch.nn as nn

from src.utils.obsUtils import GLOBAL_STATE_DIM, N_MOVE, N_ATTACK

# Joint action dim: 2 predators × (move_oh(3) + turn(1) + attack_oh(2))
PRED_ACTION_ENCODE_DIM = N_MOVE + 1 + N_ATTACK  # 6
JOINT_ACTION_DIM       = 2 * PRED_ACTION_ENCODE_DIM  # 12


def _makeMLP(inDim: int, hiddenDim: int, outDim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inDim, hiddenDim),
        nn.ReLU(),
        nn.Linear(hiddenDim, hiddenDim),
        nn.ReLU(),
        nn.Linear(hiddenDim, outDim),
    )


class TwinQNetwork(nn.Module):
    """
    Centralised twin soft Q-network.

    Parameters
    ----------
    hiddenDim : width of each MLP's hidden layers
    """

    def __init__(self, hiddenDim: int = 256):
        super().__init__()
        inDim = GLOBAL_STATE_DIM + JOINT_ACTION_DIM   # 441 + 12 = 453

        self.q1 = _makeMLP(inDim, hiddenDim, 1)
        self.q2 = _makeMLP(inDim, hiddenDim, 1)

    def forward(
        self,
        globalState:  torch.Tensor,
        jointAction:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        globalState  : (B, 441)
        jointAction  : (B, 12)
        Returns (Q1, Q2) each (B, 1)
        """
        x = torch.cat([globalState, jointAction], dim=-1)  # (B, 453)
        return self.q1(x), self.q2(x)

    def q1Value(
        self,
        globalState: torch.Tensor,
        jointAction: torch.Tensor,
    ) -> torch.Tensor:
        """Return Q1 only — used for the actor gradient."""
        x = torch.cat([globalState, jointAction], dim=-1)
        return self.q1(x)

    def minQ(
        self,
        globalState: torch.Tensor,
        jointAction: torch.Tensor,
    ) -> torch.Tensor:
        """min(Q1, Q2) — used for soft Bellman target."""
        q1, q2 = self.forward(globalState, jointAction)
        return torch.min(q1, q2)


def encodeAction(moveOH: torch.Tensor, turnCont: torch.Tensor, attackOH: torch.Tensor) -> torch.Tensor:
    """
    Encode one predator's action as a 6-d vector for Q-net input.

    moveOH   : (B, 3) — Gumbel-Softmax one-hot or hard one-hot
    turnCont : (B, 1) — continuous value in [-1, 1]
    attackOH : (B, 2) — Gumbel-Softmax one-hot or hard one-hot
    Returns  : (B, 6)
    """
    return torch.cat([moveOH, turnCont, attackOH], dim=-1)


def buildJointAction(
    moveOH0:   torch.Tensor,
    turnCont0: torch.Tensor,
    attackOH0: torch.Tensor,
    moveOH1:   torch.Tensor,
    turnCont1: torch.Tensor,
    attackOH1: torch.Tensor,
) -> torch.Tensor:
    """
    Concatenate both predators' encoded actions into the 12-d joint vector.
    Returns : (B, 12)
    """
    a0 = encodeAction(moveOH0, turnCont0, attackOH0)  # (B, 6)
    a1 = encodeAction(moveOH1, turnCont1, attackOH1)  # (B, 6)
    return torch.cat([a0, a1], dim=-1)                  # (B, 12)
