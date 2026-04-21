from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.obsUtils import N_MOVE, N_PREY_TURN
from .voxelEncoder import OUTPUT_DIM


class OMHead(nn.Module):
    def __init__(self, encoderDim: int = OUTPUT_DIM, hiddenDim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(encoderDim, hiddenDim),
            nn.ReLU(),
        )
        self.moveHead = nn.Linear(hiddenDim, N_MOVE)
        self.turnHead = nn.Linear(hiddenDim, N_PREY_TURN)

    def forward(self, encoderFeats: torch.Tensor):
        h = self.shared(encoderFeats)
        return self.moveHead(h), self.turnHead(h)

    def predictProbs(self, encoderFeats: torch.Tensor):
        moveL, turnL = self.forward(encoderFeats)
        return F.softmax(moveL, dim=-1), F.softmax(turnL, dim=-1)

    def loss(self, encoderFeats: torch.Tensor, trueMove: torch.Tensor, trueTurn: torch.Tensor) -> torch.Tensor:
        moveL, turnL = self.forward(encoderFeats)
        return F.cross_entropy(moveL, trueMove) + F.cross_entropy(turnL, trueTurn)
