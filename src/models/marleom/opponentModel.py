from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.obsUtils import OBS_DIM, N_MOVE, N_TURN, N_ATTACK


class OpponentModel(nn.Module):
    """Level model for MARLeOM prey-action prediction."""

    def __init__(self, obsDim: int = OBS_DIM, hiddenDim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obsDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
        )
        self.moveHead   = nn.Linear(hiddenDim, N_MOVE)
        self.turnHead   = nn.Linear(hiddenDim, N_TURN)
        self.attackHead = nn.Linear(hiddenDim, N_ATTACK)

    def forward(self, oppObs: torch.Tensor):
        h = self.net(oppObs)
        return self.moveHead(h), self.turnHead(h), self.attackHead(h)

    def computeLoss(self, oppObs: torch.Tensor, trueMove: torch.Tensor, trueTurn: torch.Tensor, trueAttack: torch.Tensor) -> torch.Tensor:
        moveLogits, turnLogits, attackLogits = self.forward(oppObs)
        return (
            F.cross_entropy(moveLogits, trueMove)
            + F.cross_entropy(turnLogits, trueTurn)
            + F.cross_entropy(attackLogits, trueAttack)
        )
