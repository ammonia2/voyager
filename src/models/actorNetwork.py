from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from src.utils.obsUtils import N_MOVE, N_ATTACK, OBS_DIM
from src.models.voxelEncoder import VoxelEncoder, OUTPUT_DIM
from src.models.omHead import OMHead

LOG_STD_MIN = -4.0
LOG_STD_MAX =  1.0


class ActorNetwork(nn.Module):
    """
    MAPPO actor for a single PREDATOR agent.

    Action space:
      - move:   discrete  (3 classes)
      - turn:   CONTINUOUS in [-1, 1], sampled from a Tanh-squashed Normal
      - attack: discrete  (2 classes)

    Pipeline:
      flatObs -> VoxelEncoder -> backbone -> policy heads
      flatObs -> VoxelEncoder -> OMHead  (joint training, gradients flow into encoder)
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
        self.moveHead       = nn.Linear(hiddenDim, N_MOVE)
        self.turnMeanHead   = nn.Linear(hiddenDim, 1)   # continuous turn mean
        self.turnLogStdHead = nn.Linear(hiddenDim, 1)   # continuous turn log-std
        self.attackHead     = nn.Linear(hiddenDim, N_ATTACK)

    def forward(self, flatObs: torch.Tensor):
        """
        flatObs: (B, OBS_DIM=147)
        returns:
          moveP      (B, 3)   - categorical probabilities
          turn_mean  (B, 1)   - tanh-squashed mean in (-1, 1)
          turn_std   (B, 1)   - std > 0
          attackP    (B, 2)   - categorical probabilities
        """
        enc  = self.encoder(flatObs)
        h    = self.backbone(enc)
        turn_mean   = torch.tanh(self.turnMeanHead(h))
        turn_log_std = self.turnLogStdHead(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return (
            F.softmax(self.moveHead(h),   dim=-1),
            turn_mean,
            torch.exp(turn_log_std),
            F.softmax(self.attackHead(h), dim=-1),
        )

    def sampleAction(self, flatObs: torch.Tensor):
        """
        Sample action indices/values and compute joint log-prob + entropy.
        returns: mIdx (int tensor), tCont (float tensor), aIdx (int tensor),
                 logP (scalar), entropy (scalar)
        """
        moveP, turn_mean, turn_std, attackP = self(flatObs)

        moveDist   = Categorical(moveP)
        turnDist   = Normal(turn_mean, turn_std)
        attackDist = Categorical(attackP)

        mIdx  = moveDist.sample()
        tCont = turnDist.sample().clamp(-1.0, 1.0)   # continuous value in [-1,1]
        aIdx  = attackDist.sample()

        logP = (
            moveDist.log_prob(mIdx)
            + turnDist.log_prob(tCont).sum(-1)        # sum over the 1-dim
            + attackDist.log_prob(aIdx)
        )
        entropy = (
            moveDist.entropy()
            + turnDist.entropy().sum(-1)
            + attackDist.entropy()
        )
        return mIdx, tCont, aIdx, logP, entropy

    def omLoss(
        self,
        flatObs: torch.Tensor,
        trueMove: torch.Tensor,
        trueTurn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict prey's (move, turn) from encoder features.
        Prey has no attack, so only two heads are predicted.
        Gradients propagate into the VoxelEncoder.
        """
        enc = self.encoder(flatObs)
        return self.omHead.loss(enc, trueMove, trueTurn)