"""
masac/actorNetwork.py
=====================
MASAC stochastic actor for a single PREDATOR agent.

Action space (matches mappoEnv.py)
-----------------------------------
  move   : discrete  3 classes  (move 1 / move -1 / move 0)
  turn   : CONTINUOUS in [-1, 1], sampled via Tanh-squashed Normal (reparameterised)
  attack : discrete  2 classes  (attack 1 / attack 0)

SAC requires a differentiable sample for the actor gradient update, so:
  - Discrete heads use the Gumbel-Softmax straight-through trick.
  - Continuous turn uses the reparameterisation trick (rsample + tanh).

Pipeline
--------
  flatObs (147) → VoxelEncoder (128) → backbone (128→128)
                → move head  (Gumbel-Softmax)
                → turn head  (Tanh-Normal, reparameterised)
                → attack head (Gumbel-Softmax)
  flatObs (147) → VoxelEncoder → OMHead (aux loss, gradients flow into encoder)

The encoder is SHARED between actor and OM head so that opponent-modelling
gradients improve the state representation used for action selection.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.utils.obsUtils import N_MOVE, N_ATTACK, OBS_DIM
from src.models.masac.voxelEncoder import VoxelEncoder, OUTPUT_DIM
from src.models.masac.omHead import OMHead

LOG_STD_MIN = -4.0
LOG_STD_MAX =  1.0
GUMBEL_TAU  =  1.0   # Gumbel-Softmax temperature (anneal if desired)


class ActorNetwork(nn.Module):
    """
    MASAC stochastic actor for one predator.

    Methods
    -------
    forward(flatObs)          → raw distribution parameters
    sampleAction(flatObs)     → sampled action + log-prob (for SAC update)
    deterministicAction(flatObs) → greedy action (for evaluation)
    omLoss(flatObs, ...)      → auxiliary OM cross-entropy loss
    """

    def __init__(self, hiddenDim: int = 128):
        super().__init__()
        self.encoder = VoxelEncoder()
        self.omHead  = OMHead(encoderDim=OUTPUT_DIM)

        self.backbone = nn.Sequential(
            nn.Linear(OUTPUT_DIM, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU(),
        )

        # Discrete heads
        self.moveHead   = nn.Linear(hiddenDim, N_MOVE)    # 3-class
        self.attackHead = nn.Linear(hiddenDim, N_ATTACK)  # 2-class

        # Continuous turn head (Tanh-squashed Normal)
        self.turnMeanHead   = nn.Linear(hiddenDim, 1)
        self.turnLogStdHead = nn.Linear(hiddenDim, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, flatObs: torch.Tensor):
        """
        flatObs : (B, OBS_DIM=147)
        Returns
        -------
        moveLogits   : (B, 3)
        turnMean     : (B, 1)   — pre-squash (raw Normal mean)
        turnLogStd   : (B, 1)   — clamped log-std
        attackLogits : (B, 2)
        """
        enc  = self.encoder(flatObs)           # (B, 128)
        h    = self.backbone(enc)              # (B, 128)

        moveLogits   = self.moveHead(h)
        attackLogits = self.attackHead(h)
        turnMean     = self.turnMeanHead(h)
        turnLogStd   = self.turnLogStdHead(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

        return moveLogits, turnMean, turnLogStd, attackLogits

    # ------------------------------------------------------------------
    # SAC sample — differentiable for actor gradient
    # ------------------------------------------------------------------

    def sampleAction(self, flatObs: torch.Tensor):
        """
        Reparameterised sample for SAC actor loss.

        Returns
        -------
        moveOH     : (B, 3)   Gumbel-Softmax one-hot  (differentiable)
        turnTanh   : (B, 1)   Tanh-squashed sample    (differentiable)
        attackOH   : (B, 2)   Gumbel-Softmax one-hot  (differentiable)
        logProb    : (B,)     scalar log-probability
        """
        moveLogits, turnMean, turnLogStd, attackLogits = self(flatObs)

        # --- Discrete move (Gumbel-Softmax straight-through) ---
        moveOH = F.gumbel_softmax(moveLogits, tau=GUMBEL_TAU, hard=True)

        # --- Discrete attack (Gumbel-Softmax straight-through) ---
        attackOH = F.gumbel_softmax(attackLogits, tau=GUMBEL_TAU, hard=True)

        # --- Continuous turn (reparameterised Tanh-Normal) ---
        turnDist = Normal(turnMean, torch.exp(turnLogStd))
        rawTurn  = turnDist.rsample()                           # (B, 1)
        turnTanh = torch.tanh(rawTurn)                         # squash to (-1, 1)

        # Log-prob with Tanh correction: log π(a|s) = log N(u) - log(1-tanh²(u))
        turnLogP = turnDist.log_prob(rawTurn) - torch.log(
            1.0 - turnTanh.pow(2) + 1e-6
        )                                                        # (B, 1)

        # --- Log-prob for discrete heads (CORRECTED) ---
        # For SAC with discrete actions, compute log-prob from the DISTRIBUTION, not hardened sample.
        # Use log-softmax on raw logits to get true categorical log-probs for gradient computation.
        moveLogP   = F.log_softmax(moveLogits,   dim=-1).gather(1, moveOH.argmax(1, keepdim=True))
        attackLogP = F.log_softmax(attackLogits, dim=-1).gather(1, attackOH.argmax(1, keepdim=True))

        logProb = (turnLogP + moveLogP + attackLogP).squeeze(-1)  # (B,)

        return moveOH, turnTanh, attackOH, logProb

    # ------------------------------------------------------------------
    # Inference helpers (no grad, returns numpy-friendly indices)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def deterministicAction(self, flatObs: torch.Tensor) -> tuple:
        """
        Greedy action for evaluation.
        Returns (move_idx: int, turn_cont: float, attack_idx: int)
        """
        moveLogits, turnMean, _, attackLogits = self(flatObs)
        moveIdx   = int(moveLogits.argmax(dim=-1).item())
        turnCont  = float(torch.tanh(turnMean).item())
        attackIdx = int(attackLogits.argmax(dim=-1).item())
        return moveIdx, turnCont, attackIdx

    @torch.no_grad()
    def stochasticAction(self, flatObs: torch.Tensor) -> tuple:
        """
        Stochastic action for worker rollouts (no gradient).
        Returns (move_idx: int, turn_cont: float, attack_idx: int)
        """
        moveLogits, turnMean, turnLogStd, attackLogits = self(flatObs)
        moveIdx   = int(torch.multinomial(F.softmax(moveLogits, dim=-1), 1).item())
        turnDist  = Normal(turnMean, torch.exp(turnLogStd))
        turnCont  = float(torch.tanh(turnDist.sample()).item())
        attackIdx = int(torch.multinomial(F.softmax(attackLogits, dim=-1), 1).item())
        return moveIdx, turnCont, attackIdx

    # ------------------------------------------------------------------
    # Auxiliary OM loss
    # ------------------------------------------------------------------

    def omLoss(
        self,
        flatObs:  torch.Tensor,
        trueMove: torch.Tensor,
        trueTurn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Auxiliary opponent-modelling cross-entropy loss.
        Gradients flow back through encoder → improves state representation.
        trueMove, trueTurn : (B,) long tensors of prey discrete actions.
        """
        enc = self.encoder(flatObs)
        return self.omHead.loss(enc, trueMove, trueTurn)

    def omAccuracy(self, flatObs: torch.Tensor, trueMove: torch.Tensor, trueTurn: torch.Tensor) -> float:
        """Opponent action prediction accuracy for eval logging."""
        enc = self.encoder(flatObs)
        return self.omHead.predictAccuracy(enc, trueMove, trueTurn)
