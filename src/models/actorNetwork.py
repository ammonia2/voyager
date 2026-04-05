from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.obsUtils import N_MOVE, N_TURN, N_ATTACK


class ActorNetwork(nn.Module):
    """
    Policy network with separate categorical heads for move/turn/attack.
    """

    def __init__(self, inputDim: int = 83, hiddenDim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
        )
        self.moveHead = nn.Linear(hiddenDim, N_MOVE)
        self.turnHead = nn.Linear(hiddenDim, N_TURN)
        self.attackHead = nn.Linear(hiddenDim, N_ATTACK)

    def forward(self, actorInput: torch.Tensor):
        h = self.backbone(actorInput)
        moveP = F.softmax(self.moveHead(h), dim=-1)
        turnP = F.softmax(self.turnHead(h), dim=-1)
        attackP = F.softmax(self.attackHead(h), dim=-1)
        return moveP, turnP, attackP

    def sampleAction(self, actorInput: torch.Tensor):
        moveP, turnP, attackP = self(actorInput)

        moveDist = torch.distributions.Categorical(moveP)
        turnDist = torch.distributions.Categorical(turnP)
        attackDist = torch.distributions.Categorical(attackP)

        mIdx = moveDist.sample()
        tIdx = turnDist.sample()
        aIdx = attackDist.sample()

        logP = (
            moveDist.log_prob(mIdx)
            + turnDist.log_prob(tIdx)
            + attackDist.log_prob(aIdx)
        )
        entropy = moveDist.entropy() + turnDist.entropy() + attackDist.entropy()
        return mIdx, tIdx, aIdx, logP, entropy