
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.actorNetwork import ActorNetwork
from src.models.centralisedCritic import CentralisedCritic
from src.utils.obsUtils import (
    NUM_AGENTS, PREDATOR_INDICES, PREY_INDICES,
)


def _computeGAE(
    rewards: list[float],
    values: list[float],
    dones: list[float],
    lastValue: float,
    gamma: float,
    lamda: float,
) -> list[float]:
    advantages = []
    gae        = 0.0
    nextValues = values[1:] + [lastValue]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * nextValues[t] * (1.0 - dones[t]) - values[t]
        gae   = delta + gamma * lamda * (1.0 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages


class ValueNorm(nn.Module):
    """
    Tracks running mean and running variance of targets.
    Normalizes targets for critic training and denormalizes predictions for GAE.
    """
    def __init__(self, input_shape=1):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_var", torch.ones(input_shape))
        self.register_buffer("count", torch.zeros(1))

    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.numel() / x.shape[-1] if x.ndim > 1 else x.numel()

        tot_count = self.count + batch_count
        if tot_count == 0: return

        delta = batch_mean - self.running_mean
        
        self.running_mean = self.running_mean + delta * batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + (delta**2) * self.count * batch_count / tot_count
        self.running_var = m_2 / tot_count
        self.count = tot_count

    def normalize(self, x: torch.Tensor):
        std = torch.sqrt(self.running_var + 1e-5)
        return (x - self.running_mean) / std

    def denormalize(self, x: torch.Tensor):
        std = torch.sqrt(self.running_var + 1e-5)
        return x * std + self.running_mean


class MAPPO:
    """
    MAPPO with:
    - Shared actor: VoxelEncoder (5x5x5 CNN) + OMHead trained jointly (used by all predators)
    - Centralised critic over global state
    - PPO-clip + entropy bonus + cross-entropy OM loss
    - Prey: independent PPO with its own ActorNetwork (no OM head)

    Gradient flow is separated from optimiser steps so distributed workers
    can all-reduce gradients between backward() and step():
        agent.computeGradients(rollout)    <- backward only
        <dist.all_reduce each param.grad>
        agent.applyGradients()             <- clip + step

    For single-worker use agent.update() which wraps both.
    """

    def __init__(
        self,
        lr: float           = 3e-4,
        gamma: float        = 0.99,
        lamda: float        = 0.95,
        clipEps: float      = 0.2,
        entropyCoeff: float = 0.01,
        valCoeff: float     = 0.5,
        omCoeff: float      = 0.5,
        ppoEpochs: int      = 4,
        miniBatchSize: int  = 64,
        device: str         = "cpu",
    ):
        self.gamma         = gamma
        self.lamda         = lamda
        self.clipEps       = clipEps
        self.entropyCoeff  = entropyCoeff
        self.valCoeff      = valCoeff
        self.omCoeff       = omCoeff
        self.ppoEpochs     = ppoEpochs
        self.miniBatchSize = miniBatchSize
        self.device        = torch.device(device)
        self.nPredators    = len(PREDATOR_INDICES)

        self.actor = ActorNetwork().to(self.device)  # Shared across all predators
        self.critic    = CentralisedCritic().to(self.device)
        self.preyActor = ActorNetwork().to(self.device)
        self.valueNorm = ValueNorm().to(self.device)

        self.actorOptimiser  = torch.optim.Adam(self.actor.parameters(),     lr=lr)
        self.criticOptimiser = torch.optim.Adam(self.critic.parameters(),    lr=lr)
        self.preyOptimiser   = torch.optim.Adam(self.preyActor.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def selectActions(
        self, flatObsAll: np.ndarray, explore: bool = True
    ) -> list[tuple[int, int, int]]:
        """flatObsAll: (NUM_AGENTS, OBS_DIM) -> list of (move, turn, attack) per agent"""
        actions = [(2, 2, 1)] * NUM_AGENTS
        obsT    = torch.FloatTensor(flatObsAll).to(self.device)

        for localIdx, agentIdx in enumerate(PREDATOR_INDICES):
            obs = obsT[agentIdx].unsqueeze(0)
            if explore:
                m, t, a, _, _ = self.actor.sampleAction(obs)
            else:
                moveP, turnP, attackP = self.actor(obs)
                m = moveP.argmax(-1); t = turnP.argmax(-1); a = attackP.argmax(-1)
            actions[agentIdx] = (m.item(), t.item(), a.item())

        return actions

    @torch.no_grad()
    def selectPreyAction(
        self, preyObs: np.ndarray, explore: bool = True
    ) -> tuple[int, int, int]:
        """preyObs: (OBS_DIM,)"""
        obsT = torch.FloatTensor(preyObs).unsqueeze(0).to(self.device)
        if explore:
            m, t, a, _, _ = self.preyActor.sampleAction(obsT)
        else:
            moveP, turnP, attackP = self.preyActor(obsT)
            m = moveP.argmax(-1); t = turnP.argmax(-1); a = attackP.argmax(-1)
        return (m.item(), t.item(), a.item())

    @torch.no_grad()
    def getValue(self, globalState: np.ndarray) -> float:
        """Centralised critic value for predator GAE bootstrap."""
        st = torch.FloatTensor(globalState).unsqueeze(0).to(self.device)
        return self.valueNorm.denormalize(self.critic(st).squeeze(-1))[0].item()

    # ------------------------------------------------------------------
    # Gradient computation (no optimiser step)
    # ------------------------------------------------------------------

    def computeGradients(self, rollout: dict) -> dict:
        """
        Full PPO epochs backward pass for predator actors + centralised critic.
        Zeroes gradients first. Does NOT step optimisers.
        Returns loss metrics.
        """
        T            = len(rollout["dones"])
        flatObsAll   = rollout["flatObsAll"]    # (T, NUM_AGENTS, OBS_DIM)
        globalStates = rollout["globalStates"]  # (T, GLOBAL_STATE_DIM)
        actions      = rollout["actions"]       # (T, NUM_AGENTS, 3)
        rewards      = rollout["rewards"]       # (T, NUM_AGENTS)
        dones        = rollout["dones"]         # (T,)
        oldLogProbs  = rollout["logProbs"]      # (T, nPredators)
        values       = rollout["values"]        # (T,)
        lastValue    = rollout["lastValue"]

        # Mean predator reward for shared advantage
        predRewards = [
            sum(rewards[t, i] for i in PREDATOR_INDICES) / self.nPredators
            for t in range(T)
        ]
        advArr = np.array(
            _computeGAE(predRewards, list(values), list(dones), lastValue, self.gamma, self.lamda),
            dtype=np.float32,
        )
        retArr = advArr + values
        
        # Normalise returns for the critic target
        retTRaw = torch.FloatTensor(retArr).to(self.device)
        self.valueNorm.update(retTRaw)
        retT = self.valueNorm.normalize(retTRaw)

        advArr = (advArr - advArr.mean()) / (advArr.std() + 1e-8)

        stateT       = torch.FloatTensor(globalStates).to(self.device)
        advT         = torch.FloatTensor(advArr).to(self.device)
        actT         = torch.LongTensor(actions).to(self.device)
        oldLogProbsT = torch.FloatTensor(oldLogProbs).to(self.device)
        obsPerAgent  = [
            torch.FloatTensor(flatObsAll[:, i, :]).to(self.device)
            for i in range(NUM_AGENTS)
        ]

        preyIdx     = PREY_INDICES[0]
        preyMoveT   = actT[:, preyIdx, 0]
        preyTurnT   = actT[:, preyIdx, 1]
        preyAttackT = actT[:, preyIdx, 2]

        self.actorOptimiser.zero_grad()
        self.criticOptimiser.zero_grad()

        indices = np.arange(T)
        policyLosses, valueLosses, entropies, omLosses = [], [], [], []

        for _ in range(self.ppoEpochs):
            np.random.shuffle(indices)

            for start in range(0, T, self.miniBatchSize):
                bidx = torch.LongTensor(indices[start:start + self.miniBatchSize])

                # Critic
                vPred     = self.critic(stateT[bidx]).squeeze(-1)
                valueLoss = F.mse_loss(vPred, retT[bidx])
                (self.valCoeff * valueLoss).backward()

                # Actors + OM heads
                totalPolicy  = torch.tensor(0.0, device=self.device)
                totalEntropy = torch.tensor(0.0, device=self.device)
                totalOm      = torch.tensor(0.0, device=self.device)

                for localIdx, agentIdx in enumerate(PREDATOR_INDICES):
                    obs = obsPerAgent[agentIdx][bidx]

                    moveP, turnP, attackP = self.actor(obs)
                    moveDist   = torch.distributions.Categorical(moveP)
                    turnDist   = torch.distributions.Categorical(turnP)
                    attackDist = torch.distributions.Categorical(attackP)

                    mIdx = actT[bidx, agentIdx, 0]
                    tIdx = actT[bidx, agentIdx, 1]
                    aIdx = actT[bidx, agentIdx, 2]

                    newLogP = (
                        moveDist.log_prob(mIdx)
                        + turnDist.log_prob(tIdx)
                        + attackDist.log_prob(aIdx)
                    )
                    entropy    = (moveDist.entropy() + turnDist.entropy() + attackDist.entropy()).mean()
                    ratio      = torch.exp(newLogP - oldLogProbsT[bidx, localIdx])
                    clipped    = torch.clamp(ratio, 1.0 - self.clipEps, 1.0 + self.clipEps)
                    policyLoss = -torch.min(ratio * advT[bidx], clipped * advT[bidx]).mean()
                    omLoss     = self.actor.omLoss(
                        obs, preyMoveT[bidx], preyTurnT[bidx], preyAttackT[bidx]
                    )

                    totalPolicy  = totalPolicy  + policyLoss
                    totalEntropy = totalEntropy + entropy
                    totalOm      = totalOm      + omLoss

                actorLoss = (
                    totalPolicy
                    - self.entropyCoeff * totalEntropy
                    + self.omCoeff * totalOm
                )
                actorLoss.backward()

                policyLosses.append((totalPolicy  / self.nPredators).item())
                valueLosses.append(valueLoss.item())
                entropies.append((totalEntropy    / self.nPredators).item())
                omLosses.append((totalOm          / self.nPredators).item())

        return {
            "policyLoss": float(np.mean(policyLosses)),
            "valueLoss":  float(np.mean(valueLosses)),
            "entropy":    float(np.mean(entropies)),
            "omLoss":     float(np.mean(omLosses)),
        }

    def computePreyGradients(self, rollout: dict):
        """
        Backward pass for prey independent PPO.
        Zeroes preyOptimiser gradients first. Does NOT step.
        """
        T           = len(rollout["dones"])
        preyIdx     = PREY_INDICES[0]
        flatObsAll  = rollout["flatObsAll"]
        actions     = rollout["actions"]
        rewards     = rollout["rewards"]
        dones       = rollout["dones"]
        oldLogProbs = rollout["preyLogProbs"]  # (T,)
        values      = rollout["preyValues"]    # (T,) - from prey's own value estimates
        lastValue   = rollout["preyLastValue"]

        preyRewards = list(rewards[:, preyIdx])
        advArr = np.array(
            _computeGAE(preyRewards, list(values), list(dones), lastValue, self.gamma, self.lamda),
            dtype=np.float32,
        )
        advArr = (advArr - advArr.mean()) / (advArr.std() + 1e-8)

        preyObsT     = torch.FloatTensor(flatObsAll[:, preyIdx, :]).to(self.device)
        advT         = torch.FloatTensor(advArr).to(self.device)
        oldLogProbsT = torch.FloatTensor(oldLogProbs).to(self.device)
        actT         = torch.LongTensor(actions[:, preyIdx, :]).to(self.device)

        self.preyOptimiser.zero_grad()
        indices = np.arange(T)

        for _ in range(self.ppoEpochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.miniBatchSize):
                bidx = torch.LongTensor(indices[start:start + self.miniBatchSize])

                moveP, turnP, attackP = self.preyActor(preyObsT[bidx])
                moveDist   = torch.distributions.Categorical(moveP)
                turnDist   = torch.distributions.Categorical(turnP)
                attackDist = torch.distributions.Categorical(attackP)

                mIdx = actT[bidx, 0]; tIdx = actT[bidx, 1]; aIdx = actT[bidx, 2]
                newLogP = (
                    moveDist.log_prob(mIdx)
                    + turnDist.log_prob(tIdx)
                    + attackDist.log_prob(aIdx)
                )
                entropy    = (moveDist.entropy() + turnDist.entropy() + attackDist.entropy()).mean()
                ratio      = torch.exp(newLogP - oldLogProbsT[bidx])
                clipped    = torch.clamp(ratio, 1.0 - self.clipEps, 1.0 + self.clipEps)
                policyLoss = -torch.min(ratio * advT[bidx], clipped * advT[bidx]).mean()

                (policyLoss - self.entropyCoeff * entropy).backward()

    def applyGradients(self):
        """
        Clip all gradients and step all three optimisers.
        In distributed mode, call this AFTER all-reducing gradients across workers.
        """
        nn.utils.clip_grad_norm_(self.critic.parameters(),    max_norm=0.5)
        nn.utils.clip_grad_norm_(self.actor.parameters(),     max_norm=0.5)
        nn.utils.clip_grad_norm_(self.preyActor.parameters(), max_norm=0.5)

        self.criticOptimiser.step()
        self.actorOptimiser.step()
        self.preyOptimiser.step()

    def update(self, rollout: dict) -> dict:
        """Single-worker convenience: compute gradients then immediately apply."""
        losses = self.computeGradients(rollout)
        self.computePreyGradients(rollout)
        self.applyGradients()
        return losses

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def allParams(self):
        """Flat param iterator used for initial weight broadcast in distributed mode."""
        yield from self.critic.parameters()
        yield from self.actor.parameters()
        yield from self.preyActor.parameters()

    def save(self, path: str):
        torch.save({
            "actor":     self.actor.state_dict(),
            "critic":    self.critic.state_dict(),
            "preyActor": self.preyActor.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "preyActor" in ckpt:
            self.preyActor.load_state_dict(ckpt["preyActor"])