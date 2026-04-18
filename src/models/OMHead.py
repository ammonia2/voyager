from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.obsUtils import N_MOVE, N_TURN, N_ATTACK
from src.models.voxelEncoder import OUTPUT_DIM


class OMHead(nn.Module):
    """
    Predicts opponent's next (move, turn, attack) from VoxelEncoder features.
    Trained jointly with the policy via cross-entropy loss against ground-truth actions.
    Gradients flow back through OMHead into VoxelEncoder.
    """

    def __init__(self, encoderDim: int = OUTPUT_DIM, hiddenDim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(encoderDim, hiddenDim),
            nn.ReLU(),
        )
        self.moveHead   = nn.Linear(hiddenDim, N_MOVE)
        self.turnHead   = nn.Linear(hiddenDim, N_TURN)
        self.attackHead = nn.Linear(hiddenDim, N_ATTACK)

    def forward(self, encoderFeats: torch.Tensor):
        """
        encoderFeats: (B, encoderDim)
        returns: move logits (B, 3), turn logits (B, 3), attack logits (B, 2)
        """
        h = self.shared(encoderFeats)
        return self.moveHead(h), self.turnHead(h), self.attackHead(h)

    def predictProbs(self, encoderFeats: torch.Tensor):
        moveL, turnL, attackL = self.forward(encoderFeats)
        return F.softmax(moveL, dim=-1), F.softmax(turnL, dim=-1), F.softmax(attackL, dim=-1)

    def loss(
        self,
        encoderFeats: torch.Tensor,
        trueMove: torch.Tensor,
        trueTurn: torch.Tensor,
        trueAttack: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy across all three action heads."""
        moveL, turnL, attackL = self.forward(encoderFeats)
        return (
            F.cross_entropy(moveL,   trueMove)
            + F.cross_entropy(turnL,   trueTurn)
            + F.cross_entropy(attackL, trueAttack)
        )