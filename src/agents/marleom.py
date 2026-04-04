from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.agents.actorNetwork import ActorNetwork
from src.agents.centralisedCritic import CentralizedQNetwork
from src.agents.opponentModel import OpponentModel
from src.utils.obsUtils import (
    OBS_DIM, NUM_AGENTS, N_MOVE, N_TURN, N_ATTACK, ACTION_ONEHOT_DIM,
    PREDATOR_INDICES, PREY_INDICES,
)


class MARLeOM:
    """
    MARLeOM: MASAC with embedded level-0 opponent modeling (paper Section 4).
    Controls predators (indices 0,1). Prey policy is external (scripted or independent).
    """

    def __init__(
        self,
        lr: float = 1e-3,
        alpha: float = 0.2,
        gamma: float = 0.95,
        tau: float = 0.01,
        device: str = "cpu",
        levelMix: float = 0.5,
    ):
        self.nPredators = len(PREDATOR_INDICES)
        self.alpha      = alpha
        self.gamma      = gamma
        self.tau        = tau
        self.device     = torch.device(device)
        self.levelMix   = levelMix

        self.actors = nn.ModuleList([
            ActorNetwork().to(self.device) for _ in range(self.nPredators)
        ])

        # critics[i][j]: j in {0,1} for double-Q
        self.critics = [
            [CentralizedQNetwork().to(self.device),
             CentralizedQNetwork().to(self.device)]
            for _ in range(self.nPredators)
        ]
        self.targetCritics = [
            [CentralizedQNetwork().to(self.device),
             CentralizedQNetwork().to(self.device)]
            for _ in range(self.nPredators)
        ]
        for i in range(self.nPredators):
            for j in range(2):
                self.targetCritics[i][j].load_state_dict(self.critics[i][j].state_dict())
                for p in self.targetCritics[i][j].parameters():
                    p.requires_grad = False

        self.oppModelL0 = OpponentModel().to(self.device)
        self.oppModelL1 = OpponentModel().to(self.device)

        self.actorOptimizers = [
            torch.optim.Adam(self.actors[i].parameters(), lr=lr)
            for i in range(self.nPredators)
        ]
        criticParams = []
        for i in range(self.nPredators):
            criticParams += list(self.critics[i][0].parameters())
            criticParams += list(self.critics[i][1].parameters())
        self.criticOptimizer    = torch.optim.Adam(criticParams, lr=lr)
        self.oppModelOptimizer  = torch.optim.Adam(
            list(self.oppModelL0.parameters()) + list(self.oppModelL1.parameters()),
            lr=lr,
        )

    def _predictOpponentActionMixed(self, preyObs: torch.Tensor) -> torch.Tensor:
        """
        Mix level-0 and level-1 opponent models, then take greedy one-hot action.
        Returns shape (B, 8).
        """
        with torch.no_grad():
            m0, t0, a0 = self.oppModelL0(preyObs)
            m1, t1, a1 = self.oppModelL1(preyObs)

            w = self.levelMix
            moveP = (1.0 - w) * F.softmax(m0, dim=-1) + w * F.softmax(m1, dim=-1)
            turnP = (1.0 - w) * F.softmax(t0, dim=-1) + w * F.softmax(t1, dim=-1)
            attackP = (1.0 - w) * F.softmax(a0, dim=-1) + w * F.softmax(a1, dim=-1)

            moveOh = F.one_hot(moveP.argmax(-1), N_MOVE).float()
            turnOh = F.one_hot(turnP.argmax(-1), N_TURN).float()
            attackOh = F.one_hot(attackP.argmax(-1), N_ATTACK).float()

        return torch.cat([moveOh, turnOh, attackOh], dim=-1)

    @torch.no_grad()
    def _inferBestResponseActions(
        self,
        globalState: torch.Tensor,
        actionsAll: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Approximate prey best response by exhaustive one-step search over prey actions,
        minimizing summed predator Q-values under current critics.
        """
        B = globalState.shape[0]
        preyIdx = PREY_INDICES[0]

        bestScore = torch.full((B,), float("inf"), device=self.device)
        bestMove = torch.zeros(B, dtype=torch.long, device=self.device)
        bestTurn = torch.zeros(B, dtype=torch.long, device=self.device)
        bestAttack = torch.zeros(B, dtype=torch.long, device=self.device)

        for moveIdx in range(N_MOVE):
            for turnIdx in range(N_TURN):
                for attackIdx in range(N_ATTACK):
                    candidateActions = actionsAll.clone()
                    candidateActions[:, preyIdx, 0] = moveIdx
                    candidateActions[:, preyIdx, 1] = turnIdx
                    candidateActions[:, preyIdx, 2] = attackIdx
                    candidateOh = self._actionsToOnehot(candidateActions)

                    predatorValue = torch.zeros(B, device=self.device)
                    for localIdx in range(self.nPredators):
                        q1 = self.critics[localIdx][0](globalState, candidateOh).squeeze(-1)
                        q2 = self.critics[localIdx][1](globalState, candidateOh).squeeze(-1)
                        predatorValue = predatorValue + torch.min(q1, q2)

                    better = predatorValue < bestScore
                    bestScore = torch.where(better, predatorValue, bestScore)
                    bestMove = torch.where(better, torch.full_like(bestMove, moveIdx), bestMove)
                    bestTurn = torch.where(better, torch.full_like(bestTurn, turnIdx), bestTurn)
                    bestAttack = torch.where(better, torch.full_like(bestAttack, attackIdx), bestAttack)

        return bestMove, bestTurn, bestAttack

    # ------------------------------------------------------------------
    def _buildActorInput(
        self, agentObs: torch.Tensor, preyObs: torch.Tensor
    ) -> torch.Tensor:
        """
        agentObs: (B, OBS_DIM=75)
        preyObs:  (B, OBS_DIM=75)
        Returns:  (B, 83)
        """
        preyPred = self._predictOpponentActionMixed(preyObs)  # (B, 8)
        return torch.cat([agentObs, preyPred], dim=-1)         # (B, 83)

    def _actionsToOnehot(self, actionsAll: torch.Tensor) -> torch.Tensor:
        """
        actionsAll: (B, NUM_AGENTS=3, 3) int64
        Returns:    (B, 24)  - 3 agents * 8
        """
        parts = []
        for i in range(NUM_AGENTS):
            parts.append(torch.cat([
                F.one_hot(actionsAll[:, i, 0], N_MOVE).float(),
                F.one_hot(actionsAll[:, i, 1], N_TURN).float(),
                F.one_hot(actionsAll[:, i, 2], N_ATTACK).float(),
            ], dim=-1))
        return torch.cat(parts, dim=-1)  # (B, 24)

    def _agentActionOnehot(
        self, actionsAll: torch.Tensor, localIdx: int,
        mIdx: torch.Tensor, tIdx: torch.Tensor, aIdx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full action one-hot with predator[localIdx] replaced by new sampled actions.
        Returns (B, 24).
        """
        parts = []
        for k, agentIdx in enumerate(PREDATOR_INDICES):
            if k == localIdx:
                parts.append(torch.cat([
                    F.one_hot(mIdx, N_MOVE).float(),
                    F.one_hot(tIdx, N_TURN).float(),
                    F.one_hot(aIdx, N_ATTACK).float(),
                ], dim=-1))
            else:
                parts.append(torch.cat([
                    F.one_hot(actionsAll[:, agentIdx, 0], N_MOVE).float(),
                    F.one_hot(actionsAll[:, agentIdx, 1], N_TURN).float(),
                    F.one_hot(actionsAll[:, agentIdx, 2], N_ATTACK).float(),
                ], dim=-1))
        preyIdx = PREY_INDICES[0]
        parts.append(torch.cat([
            F.one_hot(actionsAll[:, preyIdx, 0], N_MOVE).float(),
            F.one_hot(actionsAll[:, preyIdx, 1], N_TURN).float(),
            F.one_hot(actionsAll[:, preyIdx, 2], N_ATTACK).float(),
        ], dim=-1))
        return torch.cat(parts, dim=-1)  # (B, 24)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def selectActions(self, flatObsAll: np.ndarray, explore: bool = True) -> list[tuple]:
        """
        flatObsAll: (NUM_AGENTS=3, OBS_DIM=75) numpy
        Returns list of (moveIdx, turnIdx, attackIdx) for all 3 agents.
        Prey slot [2] is filled with neutral (2,2,1) - caller overrides with prey policy.
        """
        actions = [(2, 2, 1)] * NUM_AGENTS
        obsT = torch.FloatTensor(flatObsAll).to(self.device)  # (3, 75)
        preyObs = obsT[PREY_INDICES[0]].unsqueeze(0)          # (1, 75)

        for localIdx, agentIdx in enumerate(PREDATOR_INDICES):
            agentObs = obsT[agentIdx].unsqueeze(0)  # (1, 75)
            actorIn  = self._buildActorInput(agentObs, preyObs)  # (1, 83)

            if explore:
                mIdx, tIdx, aIdx, _, _ = self.actors[localIdx].sampleAction(actorIn)
            else:
                moveP, turnP, attackP = self.actors[localIdx](actorIn)
                mIdx = moveP.argmax(-1); tIdx = turnP.argmax(-1); aIdx = attackP.argmax(-1)

            actions[agentIdx] = (mIdx.item(), tIdx.item(), aIdx.item())

        return actions

    @torch.no_grad()
    def getActionDist(self, flatObsAll: np.ndarray) -> dict:
        """
        Debug: return action probability distributions and opponent predictions.
        Returns dict with keys: actor_move_logits, opponent_pred_actions, etc.
        """
        obsT = torch.FloatTensor(flatObsAll).to(self.device)
        preyObs = obsT[PREY_INDICES[0]].unsqueeze(0)

        debug_info = {"actors": [], "opponent_pred": None}

        # Opponent model prediction
        opp_pred = self._predictOpponentActionMixed(preyObs)
        debug_info["opponent_pred"] = opp_pred[0].cpu().numpy()

        # Actor distributions
        for localIdx, agentIdx in enumerate(PREDATOR_INDICES):
            agentObs = obsT[agentIdx].unsqueeze(0)
            actorIn = self._buildActorInput(agentObs, preyObs)
            moveP, turnP, attackP = self.actors[localIdx](actorIn)

            debug_info["actors"].append({
                "moveP": moveP[0].cpu().numpy(),
                "turnP": turnP[0].cpu().numpy(),
                "attackP": attackP[0].cpu().numpy(),
            })

        return debug_info

    # ------------------------------------------------------------------
    def update(self, batch: tuple) -> dict:
        """
        batch from ReplayBuffer.sample():
          obsAll:     (B, NUM_AGENTS, OBS_DIM)
          actionsAll: (B, NUM_AGENTS, 3)
          rewardsAll: (B, NUM_AGENTS)
          nextObsAll: (B, NUM_AGENTS, OBS_DIM)
          dones:      (B, NUM_AGENTS)
        """
        obsNp, actionsNp, rewardsNp, nextObsNp, donesNp = batch

        obsAll     = torch.FloatTensor(obsNp).to(self.device)
        actionsAll = torch.LongTensor(actionsNp).to(self.device)
        rewardsAll = torch.FloatTensor(rewardsNp).to(self.device)
        nextObsAll = torch.FloatTensor(nextObsNp).to(self.device)
        donesAll   = torch.FloatTensor(donesNp).to(self.device)

        B = obsAll.shape[0]
        globalState     = obsAll.view(B, -1)       # (B, 225)
        globalNextState = nextObsAll.view(B, -1)   # (B, 225)
        actionsOh       = self._actionsToOnehot(actionsAll)  # (B, 24)

        preyIdx     = PREY_INDICES[0]
        preyObsCur  = obsAll[:, preyIdx, :]      # (B, 75)
        preyObsNext = nextObsAll[:, preyIdx, :]  # (B, 75)

        # ---- Opponent model update ----
        om0Loss = self.oppModelL0.computeLoss(
            preyObsCur,
            actionsAll[:, preyIdx, 0],
            actionsAll[:, preyIdx, 1],
            actionsAll[:, preyIdx, 2],
        )
        brMove, brTurn, brAttack = self._inferBestResponseActions(globalState, actionsAll)
        om1Loss = self.oppModelL1.computeLoss(
            preyObsCur,
            brMove,
            brTurn,
            brAttack,
        )
        omLoss = om0Loss + om1Loss
        self.oppModelOptimizer.zero_grad()
        omLoss.backward()
        self.oppModelOptimizer.step()

        # ---- Next-step actions for target Q ----
        with torch.no_grad():
            nextPredActionsOhParts = []
            nextLogProbs = torch.zeros(B, self.nPredators, device=self.device)

            for localIdx, agentIdx in enumerate(PREDATOR_INDICES):
                agentNextObs = nextObsAll[:, agentIdx, :]
                actorIn = self._buildActorInput(agentNextObs, preyObsNext)
                mIdx, tIdx, aIdx, logP, _ = self.actors[localIdx].sampleAction(actorIn)
                nextPredActionsOhParts.append(torch.cat([
                    F.one_hot(mIdx, N_MOVE).float(),
                    F.one_hot(tIdx, N_TURN).float(),
                    F.one_hot(aIdx, N_ATTACK).float(),
                ], dim=-1))
                nextLogProbs[:, localIdx] = logP

            # Use stored prey action for next step (prey is external)
            preyNextOh = torch.cat([
                F.one_hot(actionsAll[:, preyIdx, 0], N_MOVE).float(),
                F.one_hot(actionsAll[:, preyIdx, 1], N_TURN).float(),
                F.one_hot(actionsAll[:, preyIdx, 2], N_ATTACK).float(),
            ], dim=-1)

            nextActionsOh = torch.cat(nextPredActionsOhParts + [preyNextOh], dim=-1)  # (B, 24)

        # ---- Critic update (eq. 9-10) ----
        criticLoss = torch.tensor(0.0, device=self.device)
        for localIdx, agentIdx in enumerate(PREDATOR_INDICES):
            reward = rewardsAll[:, agentIdx].unsqueeze(-1)
            done   = donesAll[:, agentIdx].unsqueeze(-1)
            logP   = nextLogProbs[:, localIdx].unsqueeze(-1)

            with torch.no_grad():
                tQ = torch.min(
                    self.targetCritics[localIdx][0](globalNextState, nextActionsOh),
                    self.targetCritics[localIdx][1](globalNextState, nextActionsOh),
                )
                yTarget = reward + self.gamma * (1.0 - done) * (tQ - self.alpha * logP)

            q1 = self.critics[localIdx][0](globalState, actionsOh)
            q2 = self.critics[localIdx][1](globalState, actionsOh)
            criticLoss = criticLoss + F.mse_loss(q1, yTarget) + F.mse_loss(q2, yTarget)

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        self.criticOptimizer.step()

        # ---- Actor update (eq. 11) ----
        actorLoss = torch.tensor(0.0, device=self.device)
        for localIdx, agentIdx in enumerate(PREDATOR_INDICES):
            agentObs = obsAll[:, agentIdx, :]
            actorIn  = self._buildActorInput(agentObs, preyObsCur)
            mIdx, tIdx, aIdx, logP, _ = self.actors[localIdx].sampleAction(actorIn)

            newActionsOh = self._agentActionOnehot(actionsAll, localIdx, mIdx, tIdx, aIdx)
            minQ = torch.min(
                self.critics[localIdx][0](globalState, newActionsOh),
                self.critics[localIdx][1](globalState, newActionsOh),
            )
            actorLoss = actorLoss + (self.alpha * logP.unsqueeze(-1) - minQ).mean()

        for optim in self.actorOptimizers:
            optim.zero_grad()
        actorLoss.backward()
        for optim in self.actorOptimizers:
            optim.step()

        # ---- Soft target update (eq. 12) ----
        for localIdx in range(self.nPredators):
            for j in range(2):
                for pL, pT in zip(
                    self.critics[localIdx][j].parameters(),
                    self.targetCritics[localIdx][j].parameters(),
                ):
                    pT.data.copy_(self.tau * pL.data + (1.0 - self.tau) * pT.data)

        return {
            "criticLoss": criticLoss.item(),
            "actorLoss":  actorLoss.item(),
            "omLoss":     omLoss.item(),
            "om0Loss":    om0Loss.item(),
            "om1Loss":    om1Loss.item(),
        }

    def save(self, path: str):
        torch.save({
            "actors":   [a.state_dict() for a in self.actors],
            "critics":  [[self.critics[i][j].state_dict() for j in range(2)]
                         for i in range(self.nPredators)],
            "oppModelL0": self.oppModelL0.state_dict(),
            "oppModelL1": self.oppModelL1.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for i, a in enumerate(self.actors):
            a.load_state_dict(ckpt["actors"][i])
        for i in range(self.nPredators):
            for j in range(2):
                self.critics[i][j].load_state_dict(ckpt["critics"][i][j])
                self.targetCritics[i][j].load_state_dict(ckpt["critics"][i][j])
        if "oppModelL0" in ckpt and "oppModelL1" in ckpt:
            self.oppModelL0.load_state_dict(ckpt["oppModelL0"])
            self.oppModelL1.load_state_dict(ckpt["oppModelL1"])
        elif "oppModel" in ckpt:
            # Backward compatibility with older checkpoints.
            self.oppModelL0.load_state_dict(ckpt["oppModel"])
            self.oppModelL1.load_state_dict(ckpt["oppModel"])