from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.actorNetwork import ActorNetwork
from src.models.centralisedCritic import CentralizedQNetwork
from src.models.opponentModel import OpponentModel
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
        levelMixWarmupSteps: int = 5000,   # keep levelMix=0 until critic is meaningful
        levelMixRampSteps: int  = 10000,   # linearly ramp to 0.5 over this many steps after warmup
    ):
        self.nPredators = len(PREDATOR_INDICES)
        self.gamma      = gamma
        self.tau        = tau
        self.device     = torch.device(device)

        # ---- Auto-tuned entropy coefficient (SAC-v2 style) ----
        # ---- TES-SAC: entropy-tracking target entropy schedule ----
        # Instead of a fixed time-based annealing, we drop targetEntropy by a constant
        # factor only when the actual policy entropy has stabilized near the current target.
        # This avoids alpha collapsing to near-zero before the policy has learned anything.
        # Reference: Xu et al. 2021 "Target Entropy Annealing for Discrete SAC" (NeurIPS).
        #
        # Floor is 0.3*log|A| (not 0.01 as in the paper) — higher floor needed because
        # our reward scale (~30) is large enough to saturate the actor without entropy reg.
        _logA = np.log(N_MOVE * N_TURN * N_ATTACK)
        self.targetEntropy          = _logA              # start at max entropy per TES-SAC
        self.targetEntropyFloor     = 0.3 * _logA        # ~0.87 — prevents logit saturation
        self._entropyDropFactor     = 0.75               # multiply target by this on each drop
        self._entropyTolerance      = 0.05 * _logA       # ~0.15 — "close enough" threshold
        self._entropyStableWindow   = 200                # updates before declaring stable
        self._entropyStableCount    = 0
        self._entropyEma            = _logA              # EMA of actual policy entropy
        self.logAlpha      = torch.tensor(np.log(alpha), dtype=torch.float32,
                                          device=torch.device(device), requires_grad=True)
        self.alphaOptimizer = torch.optim.Adam([self.logAlpha], lr=lr)

        # ---- Entropy penalty coefficient (SD-SAC) ----
        # Constrains how fast the actor can change per update step.
        # Small value — just enough to damp oscillation without hurting learning.
        self.entropyPenaltyCoef = 0.1

        # Dynamic levelMix schedule: 0 during warmup, linear ramp to 0.5
        self.levelMixWarmupSteps = levelMixWarmupSteps
        self.levelMixRampSteps   = levelMixRampSteps
        self.levelMix            = 0.0   # starts at 0, updated each call to update()

        # Bayesian mixing weights (paper eq. 7-8): moving average of per-level
        # prediction accuracy.  psi[m] = decaying avg of p(m | observed action).
        # Initialized to equal weight (0.5 each).
        self._psi        = np.array([0.5, 0.5], dtype=np.float64)
        self._bayesMomentum = 0.99   # decay for moving average
        self._updateCount    = 0

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
        self.criticParams = []
        for i in range(self.nPredators):
            self.criticParams += list(self.critics[i][0].parameters())
            self.criticParams += list(self.critics[i][1].parameters())
        self.criticOptimizer    = torch.optim.Adam(self.criticParams, lr=lr)
        self.oppModelOptimizer  = torch.optim.Adam(
            list(self.oppModelL0.parameters()) + list(self.oppModelL1.parameters()),
            lr=lr,
        )

    def _updateLevelMixSchedule(self):
        """Linearly ramp levelMix from 0 to 0.5 after warmup period."""
        n = self._updateCount
        if n <= self.levelMixWarmupSteps:
            self.levelMix = 0.0
        elif n <= self.levelMixWarmupSteps + self.levelMixRampSteps:
            progress = (n - self.levelMixWarmupSteps) / self.levelMixRampSteps
            self.levelMix = 0.5 * progress
        else:
            self.levelMix = 0.5

    def _updateTargetEntropySchedule(self, currentEntropy: float):
        """
        TES-SAC: drop targetEntropy by _entropyDropFactor only when actual policy
        entropy has stabilized near the current target for _entropyStableWindow updates.
        Stops dropping at targetEntropyFloor to prevent alpha collapse.
        Gated behind levelMixWarmupSteps — no drops during critic warmup period.
        """
        # Don't adjust during critic warmup — policy is still random
        if self._updateCount <= self.levelMixWarmupSteps:
            return

        # EMA of actual policy entropy (momentum 0.95 — slow enough to detect stability)
        self._entropyEma = 0.95 * self._entropyEma + 0.05 * currentEntropy

        if abs(self._entropyEma - self.targetEntropy) < self._entropyTolerance:
            self._entropyStableCount += 1
        else:
            self._entropyStableCount = 0

        if self._entropyStableCount >= self._entropyStableWindow:
            newTarget = self.targetEntropy * self._entropyDropFactor
            self.targetEntropy = max(newTarget, self.targetEntropyFloor)
            self._entropyStableCount = 0  # reset after each step down

    def _updateBayesWeights(
        self,
        preyObsCur: torch.Tensor,
        trueMove: torch.Tensor,
        trueTurn: torch.Tensor,
        trueAttack: torch.Tensor,
    ):
        """
        Update psi (moving average of Bayesian posterior) using observed prey actions.
        Paper eq. 7: p(m | a^o) proportional to pi_m(a^o | o^o) * p(m).
        We compute this per-sample then average over the batch.
        """
        with torch.no_grad():
            m0, t0, a0 = self.oppModelL0(preyObsCur)
            m1, t1, a1 = self.oppModelL1(preyObsCur)

            # Log-likelihoods of the actual observed prey action under each level
            logLikL0 = (
                F.cross_entropy(m0, trueMove,   reduction="none") * -1 +
                F.cross_entropy(t0, trueTurn,   reduction="none") * -1 +
                F.cross_entropy(a0, trueAttack, reduction="none") * -1
            )  # (B,)
            logLikL1 = (
                F.cross_entropy(m1, trueMove,   reduction="none") * -1 +
                F.cross_entropy(t1, trueTurn,   reduction="none") * -1 +
                F.cross_entropy(a1, trueAttack, reduction="none") * -1
            )  # (B,)

            # Convert to likelihoods, weight by current psi prior
            likL0 = torch.exp(logLikL0) * self._psi[0]
            likL1 = torch.exp(logLikL1) * self._psi[1]
            denom = likL0 + likL1 + 1e-8

            # Posterior p(m | a^o), averaged over batch
            postL0 = (likL0 / denom).mean().item()
            postL1 = (likL1 / denom).mean().item()

        # Decaying moving average update (paper: use moving avg to approximate prior)
        m = self._bayesMomentum
        self._psi[0] = m * self._psi[0] + (1 - m) * postL0
        self._psi[1] = m * self._psi[1] + (1 - m) * postL1
        # Renormalize
        self._psi /= self._psi.sum()

    def _predictOpponentActionMixed(self, preyObs: torch.Tensor) -> torch.Tensor:
        """
        Mix level-0 and level-1 opponent models using dynamic weights.
        During warmup levelMix=0 so only L0 is used.
        After warmup, Bayesian psi weights blend the two levels.
        Returns shape (B, 8).
        """
        with torch.no_grad():
            m0, t0, a0 = self.oppModelL0(preyObs)

            if self.levelMix < 1e-6:
                # Pure L0 — don't even run L1 during warmup
                moveP   = F.softmax(m0, dim=-1)
                turnP   = F.softmax(t0, dim=-1)
                attackP = F.softmax(a0, dim=-1)
            else:
                m1, t1, a1 = self.oppModelL1(preyObs)
                # Use Bayesian psi weights scaled by levelMix schedule
                w0 = float(self._psi[0]) * (1.0 - self.levelMix)
                w1 = float(self._psi[1]) * self.levelMix
                wSum = w0 + w1 + 1e-8
                w0 /= wSum
                w1 /= wSum

                moveP   = w0 * F.softmax(m0, dim=-1) + w1 * F.softmax(m1, dim=-1)
                turnP   = w0 * F.softmax(t0, dim=-1) + w1 * F.softmax(t1, dim=-1)
                attackP = w0 * F.softmax(a0, dim=-1) + w1 * F.softmax(a1, dim=-1)

            moveOh   = F.one_hot(moveP.argmax(-1),   N_MOVE).float()
            turnOh   = F.one_hot(turnP.argmax(-1),   N_TURN).float()
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

        self._updateCount += 1
        self._updateLevelMixSchedule()

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

        # Update Bayesian mixing weights using actual observed prey actions
        self._updateBayesWeights(
            preyObsCur,
            actionsAll[:, preyIdx, 0],
            actionsAll[:, preyIdx, 1],
            actionsAll[:, preyIdx, 2],
        )

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

            # Use opponent model to predict prey's next action instead of
            # storing current-step prey action — fixes off-by-one Bellman bias.
            preyNextOh = self._predictOpponentActionMixed(preyObsNext)  # (B, 8)

            nextActionsOh = torch.cat(nextPredActionsOhParts + [preyNextOh], dim=-1)  # (B, 24)

        # ---- Critic update (SD-SAC: double-average Q with Q-clip) ----
        # Using average(Q1, Q2) instead of min(Q1, Q2) for the target fixes the
        # underestimation bias that causes pessimistic exploration in discrete SAC.
        # Q-clip bounds the target to the theoretically possible reward range:
        #   min possible return = -TIME_PENALTY * MAX_STEPS = -0.3 * 100 = -30
        #   max possible return = TAG_REWARD = 10
        # Add small margin for gamma discounting: clip to [-35, 12].
        qClipMin = -35.0
        qClipMax = 12.0
        alpha    = self.logAlpha.exp().detach()   # current alpha, no grad here

        criticLoss = torch.tensor(0.0, device=self.device)
        for localIdx, agentIdx in enumerate(PREDATOR_INDICES):
            reward = rewardsAll[:, agentIdx].unsqueeze(-1)
            done   = donesAll[:, agentIdx].unsqueeze(-1)
            logP   = nextLogProbs[:, localIdx].unsqueeze(-1)

            with torch.no_grad():
                tQ1 = self.targetCritics[localIdx][0](globalNextState, nextActionsOh)
                tQ2 = self.targetCritics[localIdx][1](globalNextState, nextActionsOh)
                # Average instead of min — fixes underestimation bias
                tQAvg   = (tQ1 + tQ2) / 2.0
                yTarget = reward + self.gamma * (1.0 - done) * (tQAvg - alpha * logP)
                # Clip to valid reward range — prevents Q explosion/collapse
                yTarget = torch.clamp(yTarget, qClipMin, qClipMax)

            q1 = self.critics[localIdx][0](globalState, actionsOh)
            q2 = self.critics[localIdx][1](globalState, actionsOh)
            criticLoss = criticLoss + F.mse_loss(q1, yTarget) + F.mse_loss(q2, yTarget)

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        nn.utils.clip_grad_norm_(self.criticParams, max_norm=1.0)
        self.criticOptimizer.step()

        # ---- Actor update (SD-SAC: standard loss + entropy penalty) ----
        # Entropy penalty constrains per-step policy change, breaking the unstable
        # coupling loop between actor and critic in discrete SAC.
        # We approximate KL(pi_new || uniform) as -H(pi_new), i.e. penalise
        # low-entropy policies directly — cheap and effective.
        actorLoss = torch.tensor(0.0, device=self.device)
        totalEntropy = torch.tensor(0.0, device=self.device)

        for localIdx, agentIdx in enumerate(PREDATOR_INDICES):
            agentObs = obsAll[:, agentIdx, :]
            actorIn  = self._buildActorInput(agentObs, preyObsCur)
            mIdx, tIdx, aIdx, logP, entropy = self.actors[localIdx].sampleAction(actorIn)

            newActionsOh = self._agentActionOnehot(actionsAll, localIdx, mIdx, tIdx, aIdx)
            # Use average Q (not min Q) consistent with critic update
            qAvg = (
                self.critics[localIdx][0](globalState, newActionsOh) +
                self.critics[localIdx][1](globalState, newActionsOh)
            ) / 2.0

            # Standard SAC actor loss: maximise Q - alpha * logP
            sacLoss = (alpha * logP.unsqueeze(-1) - qAvg).mean()
            # Entropy penalty (SD-SAC): penalise low entropy to prevent collapse
            entropyPenalty = -self.entropyPenaltyCoef * entropy.mean()
            actorLoss = actorLoss + sacLoss + entropyPenalty
            totalEntropy = totalEntropy + entropy.mean().detach()

        # Update target entropy schedule using actual policy entropy this step
        currentEntropy = (totalEntropy / self.nPredators).item()
        self._updateTargetEntropySchedule(currentEntropy)

        for optim in self.actorOptimizers:
            optim.zero_grad()
        actorLoss.backward()
        for actor in self.actors:
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        for optim in self.actorOptimizers:
            optim.step()

        # ---- Auto-tune alpha (SAC-v2: enforce target entropy constraint) ----
        # alpha loss = -(log_alpha * (log_pi + target_entropy))
        # log_pi = -entropy by definition, so avgLogP = -(totalEntropy / nPredators).
        # When entropy < targetEntropy → logP > -targetEntropy → loss grad increases alpha.
        # When entropy > targetEntropy → logP < -targetEntropy → loss grad decreases alpha.
        avgLogP   = -(totalEntropy / self.nPredators)
        alphaLoss = -(self.logAlpha * (avgLogP + self.targetEntropy)).mean()
        self.alphaOptimizer.zero_grad()
        alphaLoss.backward()
        self.alphaOptimizer.step()

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
            "alphaLoss":      alphaLoss.item(),
            "alpha":          self.logAlpha.exp().item(),
            "levelMix":       self.levelMix,
            "psiL0":          float(self._psi[0]),
            "psiL1":          float(self._psi[1]),
            "targetEntropy":  self.targetEntropy,
            "policyEntropy":  currentEntropy,
        }

    def save(self, path: str):
        torch.save({
            "actors":     [a.state_dict() for a in self.actors],
            "critics":    [[self.critics[i][j].state_dict() for j in range(2)]
                           for i in range(self.nPredators)],
            "oppModelL0": self.oppModelL0.state_dict(),
            "oppModelL1": self.oppModelL1.state_dict(),
            "logAlpha":   self.logAlpha.detach().cpu(),
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
            self.oppModelL0.load_state_dict(ckpt["oppModel"])
            self.oppModelL1.load_state_dict(ckpt["oppModel"])
        if "logAlpha" in ckpt:
            with torch.no_grad():
                self.logAlpha.copy_(ckpt["logAlpha"].to(self.device))