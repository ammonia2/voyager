from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from src.utils.obsUtils import N_MOVE, N_ATTACK

LOG_STD_MIN = -4.0
LOG_STD_MAX = 1.0


class ActorNetwork(nn.Module):
    """MARLeOM predator policy over flat obs + predicted prey action features."""

    def __init__(self, inputDim: int = 155, hiddenDim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
        )
        self.moveHead   = nn.Linear(hiddenDim, N_MOVE)
        self.turnMeanHead = nn.Linear(hiddenDim, 1)
        self.turnLogStdHead = nn.Linear(hiddenDim, 1)
        self.attackHead = nn.Linear(hiddenDim, N_ATTACK)

    def forward(self, actorInput: torch.Tensor):
        h = self.backbone(actorInput)
        turn_mean = torch.tanh(self.turnMeanHead(h))
        turn_log_std = self.turnLogStdHead(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return (
            F.softmax(self.moveHead(h), dim=-1),
            turn_mean,
            torch.exp(turn_log_std),
            F.softmax(self.attackHead(h), dim=-1),
        )

    def sampleAction(self, actorInput: torch.Tensor):
        moveP, turn_mean, turn_std, attackP = self(actorInput)
        moveDist   = Categorical(moveP)
        turnDist   = Normal(turn_mean, turn_std)
        attackDist = Categorical(attackP)

        mIdx  = moveDist.sample()
        tCont = turnDist.sample().clamp(-1.0, 1.0)
        aIdx  = attackDist.sample()
        logP = moveDist.log_prob(mIdx) + turnDist.log_prob(tCont).sum(-1) + attackDist.log_prob(aIdx)
        entropy = moveDist.entropy() + turnDist.entropy().sum(-1) + attackDist.entropy()
        return mIdx, tCont, aIdx, logP, entropy
