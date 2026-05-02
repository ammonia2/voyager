"""
masacAgent.py
=============
Multi-Agent Soft Actor-Critic (MASAC) trainer for the dual-predator setup.

Algorithm
---------
MASAC = SAC (Haarnoja et al. 2018) extended to two cooperative predator
agents using CTDE (Centralised Training / Decentralised Execution):
  - Each predator has its OWN actor (ActorNetwork) that executes independently.
  - A SHARED twin-Q centralised critic (TwinQNetwork) takes the global state
    + joint action as input, removing the non-stationarity problem.
  - Temperature α is auto-tuned per agent via the entropy-target objective.
  - An auxiliary Opponent Modelling head co-trains with the actor to predict
    the scripted prey policy (improves encoder representations).

PDC role in this file
---------------------
This class is the "GPU Learner" half of the Async PDC design:
  • Workers push to a shared PrioritizedReplayBuffer (handled externally).
  • The learner calls masacAgent.update(batch) on GPU.
  • After every update, the learner broadcasts new weights; workers call
    masacAgent.getActorState() to sync their local copies.

Key hyperparameters (Haarnoja et al. 2018 + MASAC adaptations)
--------------------------------------------------------------
  lr            = 3e-4   (actor + critic)
  alpha_lr      = 3e-4   (temperature)
  gamma         = 0.99
  tau           = 0.005  (polyak EMA for target networks)
  batchSize     = 256
  omCoeff       = 0.5    (weight for auxiliary OM cross-entropy loss)
  targetEntropy = -dim(A)  (automatic; roughly -(N_MOVE + 1 + N_ATTACK) ≈ -6)
"""
from __future__ import annotations
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.masac.actorNetwork import ActorNetwork
from src.models.masac.centralisedCritic import (
    TwinQNetwork, buildJointAction
)
from src.utils.obsUtils import (
    OBS_DIM, GLOBAL_STATE_DIM,
    N_MOVE, N_ATTACK,
    PREDATOR_INDICES, PREY_INDICES,
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
LR              = 3e-4
ALPHA_LR        = 3e-4
GAMMA           = 0.99
TAU             = 0.005        # soft target update rate
BATCH_SIZE      = 256
OM_COEFF        = 0.5          # weight for auxiliary OM loss
GRAD_CLIP       = 5.0

# Auto-entropy target: −|action_dim|
#   move(3) turn_cont(1) attack(2) → approx 6 components
TARGET_ENTROPY  = -(N_MOVE + 1 + N_ATTACK)   # -6.0


class MASACAgent:
    """
    MASAC GPU Learner.

    Manages two predator actors (actor0, actor1) + one shared twin-Q critic.
    Prey uses a fixed scripted policy and is NOT updated here.

    Parameters
    ----------
    device        : torch device for the learner ('cuda' or 'cpu')
    lr            : learning rate for actors and critic
    alphaLr       : learning rate for log-temperature parameters
    gamma         : discount factor
    tau           : polyak averaging coefficient for target networks
    omCoeff       : weight of the auxiliary OM loss added to the actor loss
    """

    def __init__(
        self,
        device:   str   = "cpu",
        lr:       float = LR,
        alphaLr:  float = ALPHA_LR,
        gamma:    float = GAMMA,
        tau:      float = TAU,
        omCoeff:  float = OM_COEFF,
        gradClip: float = GRAD_CLIP,
    ):
        self.device  = torch.device(device)
        self.gamma   = gamma
        self.tau     = tau
        self.omCoeff = omCoeff
        self.gradClip = gradClip

        # ── Actors (one per predator) ────────────────────────────────────
        self.actor0 = ActorNetwork().to(self.device)
        self.actor1 = ActorNetwork().to(self.device)

        # ── Shared twin-Q critic + frozen target copy ────────────────────
        self.critic       = TwinQNetwork().to(self.device)
        self.criticTarget = copy.deepcopy(self.critic).to(self.device)
        for p in self.criticTarget.parameters():
            p.requires_grad_(False)

        # ── Auto-tuned temperature (one per predator) ────────────────────
        self.logAlpha0 = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.logAlpha1 = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.targetEntropy = float(TARGET_ENTROPY)

        # ── Optimisers ───────────────────────────────────────────────────
        self.optActor0 = optim.Adam(self.actor0.parameters(),  lr=lr)
        self.optActor1 = optim.Adam(self.actor1.parameters(),  lr=lr)
        self.optCritic = optim.Adam(self.critic.parameters(),  lr=lr)
        self.optAlpha0 = optim.Adam([self.logAlpha0],          lr=alphaLr)
        self.optAlpha1 = optim.Adam([self.logAlpha1],          lr=alphaLr)

        self.totalUpdates = 0

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def alpha0(self) -> torch.Tensor:
        return self.logAlpha0.exp()

    @property
    def alpha1(self) -> torch.Tensor:
        return self.logAlpha1.exp()

    # ── Action selection (for workers, called on CPU actor copies) ───────

    @torch.no_grad()
    def selectActions(
        self,
        flatObs0: np.ndarray,
        flatObs1: np.ndarray,
        explore:  bool = True,
    ) -> tuple[tuple, tuple]:
        """
        Select actions for both predators given their flat observations.

        Parameters
        ----------
        flatObs0, flatObs1 : (OBS_DIM=147,) numpy arrays
        explore            : if True, sample stochastically; else greedy

        Returns
        -------
        action0, action1 : (move_idx: int, turn_cont: float, attack_idx: int)
        """
        obs0T = torch.FloatTensor(flatObs0).unsqueeze(0).to(self.device)
        obs1T = torch.FloatTensor(flatObs1).unsqueeze(0).to(self.device)

        if explore:
            action0 = self.actor0.stochasticAction(obs0T)
            action1 = self.actor1.stochasticAction(obs1T)
        else:
            action0 = self.actor0.deterministicAction(obs0T)
            action1 = self.actor1.deterministicAction(obs1T)

        return action0, action1

    # ── Global state construction ────────────────────────────────────────

    @staticmethod
    def buildGlobalStateTensor(
        flatObsAll: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """
        flatObsAll : (B, NUM_AGENTS, OBS_DIM)
        Returns    : (B, GLOBAL_STATE_DIM=441)
        """
        B = flatObsAll.shape[0]
        return torch.FloatTensor(flatObsAll.reshape(B, -1)).to(device)

    # ── Soft target update (polyak) ──────────────────────────────────────

    def _softUpdateTarget(self):
        for pTarget, pOnline in zip(
            self.criticTarget.parameters(), self.critic.parameters()
        ):
            pTarget.data.mul_(1.0 - self.tau).add_(self.tau * pOnline.data)

    # ── One gradient step ────────────────────────────────────────────────

    def update(
        self,
        obs0:        np.ndarray,    # (B, OBS_DIM)  predator-0 observations
        obs1:        np.ndarray,    # (B, OBS_DIM)  predator-1 observations
        preyObs:     np.ndarray,    # (B, OBS_DIM)  prey observations (for global state)
        act0:        np.ndarray,    # (B, 3)  [move_idx, turn_cont, attack_idx]
        act1:        np.ndarray,    # (B, 3)
        rew0:        np.ndarray,    # (B,)
        rew1:        np.ndarray,    # (B,)
        nextObs0:    np.ndarray,    # (B, OBS_DIM)
        nextObs1:    np.ndarray,    # (B, OBS_DIM)
        nextPreyObs: np.ndarray,    # (B, OBS_DIM)
        dones:       np.ndarray,    # (B,)
        isWeights:   np.ndarray,    # (B,)  importance-sampling weights from PER
        preyMoveTrue: np.ndarray,   # (B,)  prey's true move index (for OM loss)
        preyTurnTrue: np.ndarray,   # (B,)  prey's true turn index (for OM loss)
    ) -> dict:
        """
        Perform one MASAC gradient update step.

        Returns a dict of scalar losses for logging.
        """
        dev = self.device

        # ── Per-predator observation tensors ─────────────────────────────
        obs0T     = torch.FloatTensor(obs0).to(dev)         # (B, 147)
        obs1T     = torch.FloatTensor(obs1).to(dev)         # (B, 147)
        preyObsT  = torch.FloatTensor(preyObs).to(dev)      # (B, 147)
        nextObs0T = torch.FloatTensor(nextObs0).to(dev)     # (B, 147)
        nextObs1T = torch.FloatTensor(nextObs1).to(dev)     # (B, 147)
        nPreyObsT = torch.FloatTensor(nextPreyObs).to(dev)  # (B, 147)

        rew0T = torch.FloatTensor(rew0).to(dev)             # (B,)
        rew1T = torch.FloatTensor(rew1).to(dev)             # (B,)
        doneT = torch.FloatTensor(dones).to(dev)            # (B,)
        isWT  = torch.FloatTensor(isWeights).to(dev)        # (B,)

        preyMoveT = torch.LongTensor(preyMoveTrue).to(dev)  # (B,)
        preyTurnT = torch.LongTensor(preyTurnTrue).to(dev)  # (B,)

        # ── Stored actions → encoded for Q-network input ─────────────────
        act0T = torch.FloatTensor(act0).to(dev)             # (B, 3)
        act1T = torch.FloatTensor(act1).to(dev)             # (B, 3)

        def _storedToEncoded(actStored: torch.Tensor) -> torch.Tensor:
            """(B,3) [move_idx, turn_cont, attack_idx] → 6-d encoded."""
            moveIdx   = actStored[:, 0].long()
            turnCont  = actStored[:, 1].unsqueeze(1)                    # (B,1)
            attackIdx = actStored[:, 2].long()
            moveOH    = F.one_hot(moveIdx, N_MOVE).float()              # (B,3)
            attackOH  = F.one_hot(attackIdx, N_ATTACK).float()          # (B,2)
            return torch.cat([moveOH, turnCont, attackOH], dim=-1)      # (B,6)

        enc0Stored  = _storedToEncoded(act0T)               # (B,6)
        enc1Stored  = _storedToEncoded(act1T)               # (B,6)
        jointStored = torch.cat([enc0Stored, enc1Stored], dim=-1)       # (B,12)

        # ── Global state: all 3 agents' obs concatenated ─────────────────
        # (B, 147) × 3 → (B, 441) — correct CTDE global state
        gState     = torch.cat([obs0T, obs1T, preyObsT],       dim=-1)  # (B, 441)
        gNextState = torch.cat([nextObs0T, nextObs1T, nPreyObsT], dim=-1)  # (B, 441)

        # ────────────────────────────────────────────────────────────────
        # 1. Critic update
        # ────────────────────────────────────────────────────────────────
        with torch.no_grad():
            # Sample next actions from current actors
            nextM0, nextT0, nextA0, nextLP0 = self.actor0.sampleAction(nextObs0T)
            nextM1, nextT1, nextA1, nextLP1 = self.actor1.sampleAction(nextObs1T)

            nextJoint = buildJointAction(nextM0, nextT0, nextA0, nextM1, nextT1, nextA1)
            nextMinQ  = self.criticTarget.minQ(gNextState, nextJoint)     # (B,1)

            # Soft Bellman target (average predator rewards, one shared critic)
            avgRew   = (rew0T + rew1T) / 2.0                             # (B,)
            avgAlpha = (self.alpha0 + self.alpha1) / 2.0
            avgEntLP = (nextLP0 + nextLP1) / 2.0                         # (B,)

            bellmanTarget = (
                avgRew.unsqueeze(1)
                + self.gamma * (1.0 - doneT.unsqueeze(1))
                * (nextMinQ - avgAlpha * avgEntLP.unsqueeze(1))
            )                                                             # (B,1)

        q1, q2 = self.critic(gState, jointStored)                        # (B,1)

        # PER-weighted MSE
        td1 = (q1.squeeze(1) - bellmanTarget.squeeze(1)) ** 2            # (B,)
        td2 = (q2.squeeze(1) - bellmanTarget.squeeze(1)) ** 2
        criticLoss = (isWT * (td1 + td2)).mean()

        self.optCritic.zero_grad()
        criticLoss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradClip)
        self.optCritic.step()

        # TD errors for PER priority update (detached)
        tdErrors = ((td1 + td2) / 2.0).detach().cpu().numpy()

        # ────────────────────────────────────────────────────────────────
        # 2. Actor0 update
        # ────────────────────────────────────────────────────────────────
        m0, t0, a0, lp0 = self.actor0.sampleAction(obs0T)

        # Keep actor1's action fixed for actor0's Q evaluation
        with torch.no_grad():
            m1Fixed, t1Fixed, a1Fixed, lp1Fixed = self.actor1.sampleAction(obs1T)

        jAct0Update = buildJointAction(m0, t0, a0, m1Fixed, t1Fixed, a1Fixed)
        q0ForActor  = self.critic.q1Value(gState, jAct0Update)           # (B,1)

        actorLoss0 = (self.alpha0 * lp0 - q0ForActor.squeeze(1)).mean()

        # Auxiliary OM loss for actor0
        omLoss0 = self.actor0.omLoss(obs0T, preyMoveT, preyTurnT)
        totalActorLoss0 = actorLoss0 + self.omCoeff * omLoss0

        self.optActor0.zero_grad()
        totalActorLoss0.backward()
        nn.utils.clip_grad_norm_(self.actor0.parameters(), self.gradClip)
        self.optActor0.step()

        # ────────────────────────────────────────────────────────────────
        # 3. Actor1 update
        # ────────────────────────────────────────────────────────────────
        m1, t1, a1, lp1 = self.actor1.sampleAction(obs1T)

        with torch.no_grad():
            m0Fixed, t0Fixed, a0Fixed, lp0Fixed = self.actor0.sampleAction(obs0T)

        jAct1Update = buildJointAction(m0Fixed, t0Fixed, a0Fixed, m1, t1, a1)
        q1ForActor  = self.critic.q1Value(gState, jAct1Update)           # (B,1)

        actorLoss1 = (self.alpha1 * lp1 - q1ForActor.squeeze(1)).mean()

        omLoss1 = self.actor1.omLoss(obs1T, preyMoveT, preyTurnT)
        totalActorLoss1 = actorLoss1 + self.omCoeff * omLoss1

        self.optActor1.zero_grad()
        totalActorLoss1.backward()
        nn.utils.clip_grad_norm_(self.actor1.parameters(), self.gradClip)
        self.optActor1.step()

        # ────────────────────────────────────────────────────────────────
        # 4. Temperature update (auto-tune α)
        # ────────────────────────────────────────────────────────────────
        # NOTE: We resample actions here for entropy target. This is technically stale
        # w.r.t. the actions that were used for actor loss, but for off-policy SAC
        # this is acceptable (better than caching log-probs which could cause issues).
        with torch.no_grad():
            _, _, _, lp0 = self.actor0.sampleAction(obs0T)
            _, _, _, lp1 = self.actor1.sampleAction(obs1T)

        alphaLoss0 = -(self.logAlpha0 * (lp0 + self.targetEntropy).detach()).mean()
        alphaLoss1 = -(self.logAlpha1 * (lp1 + self.targetEntropy).detach()).mean()

        self.optAlpha0.zero_grad()
        alphaLoss0.backward()
        self.optAlpha0.step()

        self.optAlpha1.zero_grad()
        alphaLoss1.backward()
        self.optAlpha1.step()

        # ── Polyak target update ─────────────────────────────────────────
        self._softUpdateTarget()
        self.totalUpdates += 1

        return {
            "criticLoss":   criticLoss.item(),
            "actorLoss0":   actorLoss0.item(),
            "actorLoss1":   actorLoss1.item(),
            "omLoss0":      omLoss0.item(),
            "omLoss1":      omLoss1.item(),
            "alpha0":       self.alpha0.item(),
            "alpha1":       self.alpha1.item(),
            "tdErrors":     tdErrors,           # numpy (B,) — for PER update
        }

    # ── Checkpoint helpers ───────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor0":       self.actor0.state_dict(),
            "actor1":       self.actor1.state_dict(),
            "critic":       self.critic.state_dict(),
            "criticTarget": self.criticTarget.state_dict(),
            "logAlpha0":    self.logAlpha0.item(),
            "logAlpha1":    self.logAlpha1.item(),
            "optActor0":    self.optActor0.state_dict(),
            "optActor1":    self.optActor1.state_dict(),
            "optCritic":    self.optCritic.state_dict(),
            "optAlpha0":    self.optAlpha0.state_dict(),
            "optAlpha1":    self.optAlpha1.state_dict(),
            "totalUpdates": self.totalUpdates,
        }, path)
        print(f"[MASAC] Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor0.load_state_dict(ckpt["actor0"])
        self.actor1.load_state_dict(ckpt["actor1"])
        self.critic.load_state_dict(ckpt["critic"])
        self.criticTarget.load_state_dict(ckpt["criticTarget"])
        self.logAlpha0.data.fill_(ckpt["logAlpha0"])
        self.logAlpha1.data.fill_(ckpt["logAlpha1"])
        self.optActor0.load_state_dict(ckpt["optActor0"])
        self.optActor1.load_state_dict(ckpt["optActor1"])
        self.optCritic.load_state_dict(ckpt["optCritic"])
        self.optAlpha0.load_state_dict(ckpt["optAlpha0"])
        self.optAlpha1.load_state_dict(ckpt["optAlpha1"])
        self.totalUpdates = ckpt.get("totalUpdates", 0)
        print(f"[MASAC] Loaded checkpoint ← {path}")

    def getActorState(self) -> dict:
        """
        Return CPU state-dicts of both actors.
        Workers call this to sync their inference-only local copies.
        **CRITICAL FIX**: Detach and move to CPU to prevent CUDA tensor issues in mp.Queue.
        """
        return {
            "actor0": {k: v.cpu().detach() for k, v in self.actor0.state_dict().items()},
            "actor1": {k: v.cpu().detach() for k, v in self.actor1.state_dict().items()},
        }

    def allParams(self):
        """Generator over all trainable parameters (for grad all-reduce if needed)."""
        yield from self.actor0.parameters()
        yield from self.actor1.parameters()
        yield from self.critic.parameters()
