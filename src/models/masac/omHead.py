"""
masac/omHead.py
===============
Opponent-Modeling head for MASAC predator agents.

Predicts the PREY's next (move, turn) from the VoxelEncoder embedding.
Prey has no attack, so only two classification heads are trained.
Gradients flow back through this head into the VoxelEncoder, encouraging
the shared encoder to capture opponent behavioural patterns.

Consistent with src/models/OMHead.py but scoped to the MASAC model package.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.obsUtils import N_MOVE, N_PREY_TURN
from src.models.masac.voxelEncoder import OUTPUT_DIM


class OMHead(nn.Module):
    """
    Predicts prey (move, turn) from encoder features.

    Inputs
    ------
    encoderFeats : (B, OUTPUT_DIM=128)

    Outputs
    -------
    moveLogits : (B, N_MOVE=3)
    turnLogits : (B, N_PREY_TURN=3)
    """

    def __init__(self, encoderDim: int = OUTPUT_DIM, hiddenDim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(encoderDim, hiddenDim),
            nn.ReLU(),
        )
        self.moveHead = nn.Linear(hiddenDim, N_MOVE)        # 3 discrete classes
        self.turnHead = nn.Linear(hiddenDim, N_PREY_TURN)   # 3 discrete classes

    def forward(self, encoderFeats: torch.Tensor):
        """Returns (move_logits, turn_logits)."""
        h = self.shared(encoderFeats)
        return self.moveHead(h), self.turnHead(h)

    def predictProbs(self, encoderFeats: torch.Tensor):
        """Softmax probabilities for inference."""
        moveL, turnL = self.forward(encoderFeats)
        return F.softmax(moveL, dim=-1), F.softmax(turnL, dim=-1)

    def predictAccuracy(
        self,
        encoderFeats: torch.Tensor,
        trueMove: torch.Tensor,
        trueTurn: torch.Tensor,
    ) -> float:
        """
        Compute mean accuracy over move + turn predictions.
        Useful for the 'Opponent Action Prediction Accuracy' eval metric.
        """
        moveP, turnP = self.predictProbs(encoderFeats)
        movePred = moveP.argmax(dim=-1)
        turnPred = turnP.argmax(dim=-1)
        moveAcc = (movePred == trueMove).float().mean().item()
        turnAcc = (turnPred == trueTurn).float().mean().item()
        return (moveAcc + turnAcc) / 2.0

    def loss(
        self,
        encoderFeats: torch.Tensor,
        trueMove: torch.Tensor,
        trueTurn: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss over both heads. Used as auxiliary OM loss."""
        moveL, turnL = self.forward(encoderFeats)
        return (
            F.cross_entropy(moveL, trueMove)
            + F.cross_entropy(turnL, trueTurn)
        )
