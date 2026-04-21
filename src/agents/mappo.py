from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from src.models.actorNetwork import ActorNetwork
from src.models.preyActorNetwork import PreyActorNetwork
from src.models.centralisedCritic import CentralisedCritic
from src.utils.obsUtils import (
    NUM_AGENTS, PREDATOR_INDICES, PREY_INDICES,
)


def _computeGAE(
    rewards: list[float],
    values:  list[float],
    dones:   list[float],
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
    """Running mean/var normaliser for critic targets."""
    def __init__(self, input_shape=1):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_var",  torch.ones(input_shape))
        self.register_buffer("count",        torch.zeros(1))

    def update(self, x: torch.Tensor):
        batch_mean  = x.mean(dim=0)
        batch_var   = x.var(dim=0, unbiased=False)
        batch_count = x.numel() / x.shape[-1] if x.ndim > 1 else x.numel()
        tot_count   = self.count + batch_count
        if tot_count == 0:
            return
        delta            = batch_mean - self.running_mean
        self.running_mean = self.running_mean + delta * batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        self.running_var = m_2 / tot_count
        self.count       = tot_count

    def normalize(self, x: torch.Tensor):
        return (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)

    def denormalize(self, x: torch.Tensor):
        return x * torch.sqrt(self.running_var + 1e-5) + self.running_mean


class MAPPO:
    """
    MAPPO with:
    - Predator actor: VoxelEncoder + OMHead (shared across all predators)
        * move:   discrete (Categorical)
        * turn:   CONTINUOUS in [-1,1] (Normal distribution)
        * attack: discrete (Categorical)
    - Prey actor: PreyActorNetwork (independent)
        * move:   discrete (Categorical)
        * turn:   discrete (Categorical)
        * NO attack — prey never punches
    - Centralised critic over global state (predator GAE)
    - Prey reward: +0.1 per step survived, -10 when tagged (see env)

    Gradient flow is separated so distributed workers can all-reduce:
        agent.computeGradients(rollout)    <- backward only
        <dist.all_reduce each param.grad>
        agent.applyGradients()             <- clip + step
    """

    def __init__(
        self,
        lr: float           = 3e-4,
        gamma: float        = 0.99,
        lamda: float        = 0.95,
        clipEps: float      = 0.15,
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

        # Shared predator actor (continuous turn) + OM head
        self.actor     = ActorNetwork().to(self.device)
        self.critic    = CentralisedCritic().to(self.device)
        # Prey actor: move+turn discrete, NO attack
        self.preyActor = PreyActorNetwork().to(self.device)
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
    ) -> list[tuple]:
        """
        flatObsAll: (NUM_AGENTS, OBS_DIM)
        Predator action: (move_idx: int, turn_cont: float, attack_idx: int)
        Prey slot left as neutral — caller fills it via selectPreyAction().
        """
        actions = [(2, 0.0, 1)] * NUM_AGENTS          # neutral for every slot
        obsT    = torch.FloatTensor(flatObsAll).to(self.device)

        for agentIdx in PREDATOR_INDICES:
            obs = obsT[agentIdx].unsqueeze(0)
            if explore:
                mIdx, tCont, aIdx, _, _ = self.actor.sampleAction(obs)
                actions[agentIdx] = (
                    mIdx.item(),
                    float(tCont.squeeze().item()),
                    aIdx.item(),
                )
            else:
                moveP, turn_mean, _, attackP = self.actor(obs)
                m = moveP.argmax(-1).item()
                t = float(turn_mean.squeeze().item())   # deterministic: use mean
                a = attackP.argmax(-1).item()
                actions[agentIdx] = (m, t, a)

        return actions

    @torch.no_grad()
    def selectPreyAction(
        self, preyObs: np.ndarray, explore: bool = True
    ) -> tuple:
        """
        preyObs: (OBS_DIM,)
        Returns: (move_idx, turn_idx, 0)  — attack is always 0 (unused by prey)
        """
        obsT = torch.FloatTensor(preyObs).unsqueeze(0).to(self.device)
        if explore:
            mIdx, tIdx, _, _ = self.preyActor.sampleAction(obsT)
        else:
            moveP, turnP = self.preyActor(obsT)
            mIdx = moveP.argmax(-1)
            tIdx = turnP.argmax(-1)
        return (mIdx.item(), tIdx.item(), 0)

    @torch.no_grad()
    def getValue(self, globalState: np.ndarray) -> float:
        """Centralised critic value for predator GAE bootstrap."""
        st = torch.FloatTensor(globalState).unsqueeze(0).to(self.device)
        return self.valueNorm.denormalize(self.critic(st).squeeze(-1))[0].item()

    def applyGradients(self):
        """Clip all gradients and step all optimisers."""
        nn.utils.clip_grad_norm_(self.critic.parameters(),    max_norm=0.5)
        nn.utils.clip_grad_norm_(self.actor.parameters(),     max_norm=0.5)
        nn.utils.clip_grad_norm_(self.preyActor.parameters(), max_norm=0.5)

        self.criticOptimiser.step()
        self.actorOptimiser.step()
        self.preyOptimiser.step()

    def update(self, rollout: dict, reduceFn=None) -> dict:
        """
        Unified PPO update loop for Predator and Prey.
        Loops through PPO_EPOCHS and MINIBATCHES. In distributed mode,
        reduceFn handles gradient sync BEFORE stepping the optimiser.
        """
        T            = len(rollout["dones"])
        flatObsAll   = rollout["flatObsAll"]
        globalStates = rollout["globalStates"]
        actions      = rollout["actions"]
        rewards      = rollout["rewards"]
        dones        = rollout["dones"]
        
        # ---------------- Predator GAE ----------------
        oldLogProbs  = rollout["logProbs"]
        values       = rollout["values"]
        lastValue    = rollout["lastValue"]
        predRewards = [
            sum(rewards[t, i] for i in PREDATOR_INDICES) / self.nPredators
            for t in range(T)
        ]
        advArr = np.array(_computeGAE(predRewards, list(values), list(dones), lastValue, self.gamma, self.lamda), dtype=np.float32)
        retArr = advArr + values
        retTRaw = torch.FloatTensor(retArr).to(self.device)
        self.valueNorm.update(retTRaw)
        retT   = self.valueNorm.normalize(retTRaw)
        advArr = (advArr - advArr.mean()) / (advArr.std() + 1e-8)

        # ---------------- Prey GAE ----------------
        preyIdx         = PREY_INDICES[0]
        oldPreyLogProbs = rollout["preyLogProbs"]
        preyValues      = rollout["preyValues"]
        preyLastValue   = rollout["preyLastValue"]
        preyRewardsRaw  = list(rewards[:, preyIdx])
        preyAdvRaw      = np.array(_computeGAE(preyRewardsRaw, list(preyValues), list(dones), preyLastValue, self.gamma, self.lamda), dtype=np.float32)
        preyAdvMse      = float(np.mean(np.square(preyAdvRaw)))
        preyAdvArr      = preyAdvRaw
        preyAdvArr      = (preyAdvArr - preyAdvArr.mean()) / (preyAdvArr.std() + 1e-8)

        # ---------------- Tensors ----------------
        stateT           = torch.FloatTensor(globalStates).to(self.device)
        advT             = torch.FloatTensor(advArr).to(self.device)
        preyAdvT         = torch.FloatTensor(preyAdvArr).to(self.device)
        actsT            = torch.FloatTensor(actions).to(self.device)
        oldLogProbsT     = torch.FloatTensor(oldLogProbs).to(self.device)
        oldPreyLogProbsT = torch.FloatTensor(oldPreyLogProbs).to(self.device)
        
        obsPerAgent  = [
            torch.FloatTensor(flatObsAll[:, i, :]).to(self.device)
            for i in range(NUM_AGENTS)
        ]
        preyMoveT = actsT[:, preyIdx, 0].long()
        preyTurnT = actsT[:, preyIdx, 1].long()

        policyLosses, valueLosses, entropies, omLosses, preyEntropies = [], [], [], [], []
        indices = np.arange(T)

        for _ in range(self.ppoEpochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.miniBatchSize):
                bidx = torch.LongTensor(indices[start:start + self.miniBatchSize])

                self.actorOptimiser.zero_grad()
                self.criticOptimiser.zero_grad()
                self.preyOptimiser.zero_grad()

                # --- 1. Centralised critic ---
                vPred     = self.critic(stateT[bidx]).squeeze(-1)
                valueLoss = F.mse_loss(vPred, retT[bidx])
                (self.valCoeff * valueLoss).backward()

                # --- 2. Predator actors + OM heads ---
                totalPolicy  = torch.tensor(0.0, device=self.device)
                totalEntropy = torch.tensor(0.0, device=self.device)
                totalOm      = torch.tensor(0.0, device=self.device)

                for localIdx, agentIdx in enumerate(PREDATOR_INDICES):
                    obs = obsPerAgent[agentIdx][bidx]

                    moveP, turn_mean, turn_std, attackP = self.actor(obs)
                    moveDist   = Categorical(moveP)
                    turnDist   = Normal(turn_mean, turn_std)
                    attackDist = Categorical(attackP)

                    mIdx = actsT[bidx, agentIdx, 0].long()
                    tVal = actsT[bidx, agentIdx, 1].unsqueeze(-1)
                    aIdx = actsT[bidx, agentIdx, 2].long()

                    newLogP = (
                        moveDist.log_prob(mIdx)
                        + turnDist.log_prob(tVal).sum(-1)
                        + attackDist.log_prob(aIdx)
                    )
                    entropy = (moveDist.entropy() + turnDist.entropy().sum(-1) + attackDist.entropy()).mean()

                    ratio      = torch.exp(newLogP - oldLogProbsT[bidx, localIdx])
                    clipped    = torch.clamp(ratio, 1.0 - self.clipEps, 1.0 + self.clipEps)
                    policyLoss = -torch.min(ratio * advT[bidx], clipped * advT[bidx]).mean()

                    omLoss = self.actor.omLoss(obs, preyMoveT[bidx], preyTurnT[bidx])

                    totalPolicy  = totalPolicy  + policyLoss
                    totalEntropy = totalEntropy + entropy
                    totalOm      = totalOm      + omLoss

                actorLoss = (totalPolicy - self.entropyCoeff * totalEntropy + self.omCoeff * totalOm)
                actorLoss.backward()

                # --- 3. Prey Actor ---
                preyObsT_batch = obsPerAgent[preyIdx][bidx]
                pMoveP, pTurnP = self.preyActor(preyObsT_batch)
                pMoveDist = Categorical(pMoveP)
                pTurnDist = Categorical(pTurnP)

                pmIdx = actsT[bidx, preyIdx, 0].long()
                ptIdx = actsT[bidx, preyIdx, 1].long()

                newPreyLogP = pMoveDist.log_prob(pmIdx) + pTurnDist.log_prob(ptIdx)
                preyEntropy = (pMoveDist.entropy() + pTurnDist.entropy()).mean()
                preyRatio   = torch.exp(newPreyLogP - oldPreyLogProbsT[bidx])
                preyClipped = torch.clamp(preyRatio, 1.0 - self.clipEps, 1.0 + self.clipEps)
                preyPolicyLoss = -torch.min(preyRatio * preyAdvT[bidx], preyClipped * preyAdvT[bidx]).mean()

                (preyPolicyLoss - self.entropyCoeff * preyEntropy).backward()

                # --- 4. Sync & Step ---
                if reduceFn is not None:
                    reduceFn(list(self.allParams()))
                self.applyGradients()

                # --- Logging ---
                policyLosses.append((totalPolicy  / self.nPredators).item())
                valueLosses.append(valueLoss.item())
                entropies.append((totalEntropy    / self.nPredators).item())
                omLosses.append((totalOm          / self.nPredators).item())
                preyEntropies.append(preyEntropy.item())

        return {
            "policyLoss": float(np.mean(policyLosses)),
            "valueLoss":  float(np.mean(valueLosses)),
            "entropy":    float(np.mean(entropies)),
            "omLoss":     float(np.mean(omLosses)),
            "preyEntropy": float(np.mean(preyEntropies)) if preyEntropies else 0.0,
            "preyAdvMse": preyAdvMse,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def allParams(self):
        """Flat param iterator for initial weight broadcast in distributed mode."""
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