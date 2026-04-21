from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.obsUtils import N_MOVE, N_PREY_TURN


class PreyActorNetwork(nn.Module):
    def __init__(self, inputDim: int = 153, hiddenDim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
        )
        self.moveHead = nn.Linear(hiddenDim, N_MOVE)
        self.turnHead = nn.Linear(hiddenDim, N_PREY_TURN)

    def forward(self, preyInput: torch.Tensor):
        h = self.net(preyInput)
        return F.softmax(self.moveHead(h), dim=-1), F.softmax(self.turnHead(h), dim=-1)

    def sampleAction(self, preyInput: torch.Tensor):
        moveP, turnP = self(preyInput)
        moveDist = torch.distributions.Categorical(moveP)
        turnDist = torch.distributions.Categorical(turnP)
        mIdx = moveDist.sample()
        tIdx = turnDist.sample()
        logP = moveDist.log_prob(mIdx) + turnDist.log_prob(tIdx)
        entropy = moveDist.entropy() + turnDist.entropy()
        return mIdx, tIdx, logP, entropy
