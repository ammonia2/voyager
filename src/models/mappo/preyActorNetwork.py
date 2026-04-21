from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.obsUtils import N_MOVE, N_PREY_TURN, OBS_DIM
from .voxelEncoder import VoxelEncoder, OUTPUT_DIM


class PreyActorNetwork(nn.Module):
    def __init__(self, hiddenDim: int = 128):
        super().__init__()
        self.encoder = VoxelEncoder()

        self.backbone = nn.Sequential(
            nn.Linear(OUTPUT_DIM, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
        )
        self.moveHead = nn.Linear(hiddenDim, N_MOVE)
        self.turnHead = nn.Linear(hiddenDim, N_PREY_TURN)

    def forward(self, flatObs: torch.Tensor):
        enc = self.encoder(flatObs)
        h   = self.backbone(enc)
        return (
            F.softmax(self.moveHead(h), dim=-1),
            F.softmax(self.turnHead(h), dim=-1),
        )

    def sampleAction(self, flatObs: torch.Tensor):
        moveP, turnP = self(flatObs)
        moveDist = torch.distributions.Categorical(moveP)
        turnDist = torch.distributions.Categorical(turnP)
        mIdx = moveDist.sample()
        tIdx = turnDist.sample()
        logP = moveDist.log_prob(mIdx) + turnDist.log_prob(tIdx)
        entropy = moveDist.entropy() + turnDist.entropy()
        return mIdx, tIdx, logP, entropy
