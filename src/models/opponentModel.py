from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.obsUtils import OBS_DIM, N_MOVE, N_TURN, N_ATTACK


class OpponentModel(nn.Module):
    """
    Level-0 opponent model (paper eq. 4).
    Trained by maximum likelihood on (o_opponent, a_opponent) pairs from replay buffer.
    Predicts opponent's action distribution given their observation.
    Input:  opponent obs  (B, OBS_DIM)
    Output: move logits   (B, N_MOVE)
            turn logits   (B, N_TURN)
            attack logits (B, N_ATTACK)
    """

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

    def predictActionOnehot(self, oppObs: torch.Tensor) -> torch.Tensor:
        """
        Greedy prediction as concatenated one-hot vector of shape (B, 8).
        Used to feed predicted opponent actions into the actor at execution time.
        """
        with torch.no_grad():
            moveLogits, turnLogits, attackLogits = self.forward(oppObs)
            moveOh   = F.one_hot(moveLogits.argmax(-1),   N_MOVE).float()
            turnOh   = F.one_hot(turnLogits.argmax(-1),   N_TURN).float()
            attackOh = F.one_hot(attackLogits.argmax(-1), N_ATTACK).float()
        return torch.cat([moveOh, turnOh, attackOh], dim=-1)  # (B, 8)

    def computeLoss(
        self,
        oppObs: torch.Tensor,
        trueMove: torch.Tensor,
        trueTurn: torch.Tensor,
        trueAttack: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss across all three action heads (paper eq. 4)."""
        moveLogits, turnLogits, attackLogits = self.forward(oppObs)
        loss = (
            F.cross_entropy(moveLogits,   trueMove)
            + F.cross_entropy(turnLogits,   trueTurn)
            + F.cross_entropy(attackLogits, trueAttack)
        )
        return loss