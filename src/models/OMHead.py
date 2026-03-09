from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# matching malmoEnv.py action space
N_MOVE   = 3
N_TURN   = 3
N_ATTACK = 2


class OMHead(nn.Module):
    """
    Predicts opponent's next action distribution from encoder features.
    Sits on top of VoxelEncoder output.
    Separate heads for each action dimension (multidiscrete).
    """
    def __init__(self, encoderDim: int = 128, hiddenDim: int = 64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(encoderDim, hiddenDim),
            nn.ReLU(),
        )

        # Separate prediction head per action dimension
        self.moveHead   = nn.Linear(hiddenDim, N_MOVE)
        self.turnHead   = nn.Linear(hiddenDim, N_TURN) # could try continuous
        self.attackHead = nn.Linear(hiddenDim, N_ATTACK)

    def forward(self, encoderFeats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        encoderFeats: (B, encoderDim)
        returns: move logits (B, 3), turn logits (B, 3), attack logits (B, 2)
        """
        h = self.shared(encoderFeats)
        return self.moveHead(h), self.turnHead(h), self.attackHead(h)

    def predictProbs(self, encoderFeats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns softmax probabilities instead of logits."""
        moveL, turnL, attackL = self.forward(encoderFeats)
        return F.softmax(moveL, dim=-1), F.softmax(turnL, dim=-1), F.softmax(attackL, dim=-1)

    def loss(self, encoderFeats: torch.Tensor,
             trueMove: torch.Tensor, trueTurn: torch.Tensor, trueAttack: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss across all three action heads.
        trueMove/trueTurn/trueAttack: (B,) int class indices
        """
        moveL, turnL, attackL = self.forward(encoderFeats)
        moveLoss   = F.cross_entropy(moveL, trueMove)
        turnLoss   = F.cross_entropy(turnL, trueTurn)
        attackLoss = F.cross_entropy(attackL, trueAttack)
        return moveLoss + turnLoss + attackLoss