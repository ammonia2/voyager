from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.obsUtils import N_MOVE, N_TURN, N_ATTACK, OBS_DIM
from src.models.voxelEncoder import VoxelEncoder, OUTPUT_DIM
from src.models.omHead import OMHead


class ActorNetwork(nn.Module):
    """
    MAPPO actor for a single predator agent.
    Pipeline: flatObs -> VoxelEncoder -> policy heads
              flatObs -> VoxelEncoder -> OMHead (joint training)
    The OMHead gradients flow back through the shared encoder.
    """

    def __init__(self, hiddenDim: int = 128):
        super().__init__()
        self.encoder = VoxelEncoder()
        self.omHead  = OMHead(encoderDim=OUTPUT_DIM)

        self.backbone = nn.Sequential(
            nn.Linear(OUTPUT_DIM, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
        )
        self.moveHead   = nn.Linear(hiddenDim, N_MOVE)
        self.turnHead   = nn.Linear(hiddenDim, N_TURN)
        self.attackHead = nn.Linear(hiddenDim, N_ATTACK)

    def forward(self, flatObs: torch.Tensor):
        """
        flatObs: (B, OBS_DIM=151)
        returns: moveP (B, 3), turnP (B, 3), attackP (B, 2)
        """
        enc = self.encoder(flatObs)
        h   = self.backbone(enc)
        return (
            F.softmax(self.moveHead(h),   dim=-1),
            F.softmax(self.turnHead(h),   dim=-1),
            F.softmax(self.attackHead(h), dim=-1),
        )

    def sampleAction(self, flatObs: torch.Tensor):
        """
        Sample action indices and compute log-prob + entropy.
        returns: mIdx, tIdx, aIdx, logP (scalar), entropy (scalar)
        """
        moveP, turnP, attackP = self(flatObs)

        moveDist   = torch.distributions.Categorical(moveP)
        turnDist   = torch.distributions.Categorical(turnP)
        attackDist = torch.distributions.Categorical(attackP)

        mIdx = moveDist.sample()
        tIdx = turnDist.sample()
        aIdx = attackDist.sample()

        logP    = moveDist.log_prob(mIdx) + turnDist.log_prob(tIdx) + attackDist.log_prob(aIdx)
        entropy = moveDist.entropy() + turnDist.entropy() + attackDist.entropy()
        return mIdx, tIdx, aIdx, logP, entropy

    def omLoss(
        self,
        flatObs: torch.Tensor,
        trueMove: torch.Tensor,
        trueTurn: torch.Tensor,
        trueAttack: torch.Tensor,
    ) -> torch.Tensor:
        """Compute OM prediction loss; gradients propagate into the encoder."""
        enc = self.encoder(flatObs)
        return self.omHead.loss(enc, trueMove, trueTurn, trueAttack)