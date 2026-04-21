from __future__ import annotations
import time
import random
import os
import sys
import math
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.marleomEnv import MalmoEnv, MOVE_CMDS, TURN_CMDS, ATTACK_CMDS
from src.models.marleom.preyActorNetwork import PreyActorNetwork
from src.agents.marleom import MARLeOM
from src.utils.logs import MARLLogger
from src.utils.replayBuffer import ReplayBuffer
from src.utils.obsUtils import (
    ACTION_OH_DIM,
    flattenObsAll,
    NUM_AGENTS,
    PREDATOR_INDICES,
    PREY_INDICES,
)
MISSION_XML     = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")

# ---- Hyperparameters ----
MAX_STEPS       = 500
BUFFER_CAPACITY = 100_000
BATCH_SIZE      = 256
WARMUP_STEPS    = 5000
UPDATE_EVERY    = 4
WORLD_RESET_EVERY_UPDATES = 50
LR              = 1e-3
ALPHA           = 0.5          # higher entropy coeff forces exploration under sparse rewards
GAMMA           = 0.99         # higher discount so distant tags are worth chasing
TAU             = 0.01
DEVICE          = "cpu"
SAVE_EVERY      = 10
CKPT_DIR        = "checkpoints"
LOG_INTERVAL    = 10

# Debug flag removed — levelMix is now a dynamic schedule inside MARLeOM

NEUTRAL_ACTION  = (2, 2, 1)


def randomAction() -> tuple[int, int, int]:
    return (
        random.randint(0, len(MOVE_CMDS) - 1),
        random.uniform(-1.0, 1.0),
        random.randint(0, len(ATTACK_CMDS) - 1),
    )


def _restoreEpisodeCountersFromJsonl(logPath: str) -> tuple[int, int]:
    """Restore (episodes, wins) from existing JSONL episode records."""
    if not os.path.exists(logPath):
        return 0, 0

    episodes = 0
    wins = 0
    with open(logPath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type") != "episode":
                continue
            episodes += 1
            wins += int(rec.get("win", 0))
    return episodes, wins


# Safe zone inside the arena walls — prey steers away if it gets this close
_WALL_MARGIN = 3.0
_ARENA_MIN   = 2.0
_ARENA_MAX   = 18.0


def _wallRepulsion(px: float, pz: float) -> tuple[float, float]:
    """
    Returns a repulsion vector (dx, dz) pushing prey away from whichever
    walls it's close to. Zero when safely inside the margin.
    """
    dx, dz = 0.0, 0.0
    if px - _ARENA_MIN < _WALL_MARGIN:
        dx += (_WALL_MARGIN - (px - _ARENA_MIN))
    if _ARENA_MAX - px < _WALL_MARGIN:
        dx -= (_WALL_MARGIN - (_ARENA_MAX - px))
    if pz - _ARENA_MIN < _WALL_MARGIN:
        dz += (_WALL_MARGIN - (pz - _ARENA_MIN))
    if _ARENA_MAX - pz < _WALL_MARGIN:
        dz -= (_WALL_MARGIN - (_ARENA_MAX - pz))
    return dx, dz


def preyPolicy(preyObs: dict, predObs: list[dict]) -> tuple[int, int, int]:
    """
    Scripted prey: flee from nearest predator, with wall repulsion so it
    doesn't get pinned in corners. Never attacks.

    Direction is a weighted blend of:
      - flee vector (away from nearest predator)
      - wall repulsion vector (away from nearby walls)
    """
    preyPos = preyObs["pos"]
    preyYaw = preyObs["yaw"]
    px, pz  = float(preyPos[0]), float(preyPos[1])

    # Flee vector: away from nearest predator
    flee_dx, flee_dz = 0.0, 0.0
    nearest_dist = float("inf")
    for pObs in predObs:
        dx = px - float(pObs["pos"][0])   # away from predator
        dz = pz - float(pObs["pos"][1])
        dist = (dx ** 2 + dz ** 2) ** 0.5
        if dist < nearest_dist:
            nearest_dist = dist
            flee_dx, flee_dz = dx, dz

    # Normalise flee vector
    flee_len = (flee_dx ** 2 + flee_dz ** 2) ** 0.5
    if flee_len > 0.01:
        flee_dx /= flee_len
        flee_dz /= flee_len

    # Wall repulsion vector
    wall_dx, wall_dz = _wallRepulsion(px, pz)
    wall_len = (wall_dx ** 2 + wall_dz ** 2) ** 0.5
    if wall_len > 0.01:
        wall_dx /= wall_len
        wall_dz /= wall_len

    # Blend: wall repulsion dominates when near a wall
    near_wall = wall_len > 0.01
    if near_wall:
        goal_dx = 0.4 * flee_dx + 0.6 * wall_dx
        goal_dz = 0.4 * flee_dz + 0.6 * wall_dz
    else:
        goal_dx, goal_dz = flee_dx, flee_dz

    if (goal_dx ** 2 + goal_dz ** 2) < 0.01:
        return (0, random.randint(0, 1), 0)

    # Target angle in Malmo convention (atan2(x, z))
    target_angle = math.degrees(math.atan2(goal_dx, goal_dz)) % 360.0
    delta = (target_angle - preyYaw + 360.0) % 360.0

    if delta < 30.0 or delta > 330.0:
        turn = 2   # on target, go straight
    elif delta <= 180.0:
        turn = 1   # turn right
    else:
        turn = 0   # turn left

    return (0, turn, 0)   # move forward, computed turn, never attack


def diagnoseActionCollapse(flatObs: np.ndarray, agent: MARLeOM) -> str:
    """
    Check if predator actions have collapsed to degenerate pattern.
    Returns diagnostic string.
    """
    debug_info = agent.getActionDist(flatObs)
    
    diagnostics = []
    
    # Check opponent model prediction
    opp_pred = debug_info["opponent_pred"]
    pred_move = opp_pred[:1]  # First action is move
    prey_predicted_static = (opp_pred[0] > 0.7)  # One-hot or near one-hot on same action
    
    # Check actor move probabilities
    for idx, actor_dist in enumerate(debug_info["actors"]):
        moveP = actor_dist["moveP"]
        attackP = actor_dist["attackP"]
        
        # Calculate entropy as proxy for diversity
        move_entropy = -np.sum(moveP[moveP > 0] * np.log(moveP[moveP > 0] + 1e-8))
        attack_entropy = -np.sum(attackP[attackP > 0] * np.log(attackP[attackP > 0] + 1e-8))
        
        is_collapsed = (move_entropy < 0.1) and (attackP[-1] > 0.8)
        diagnostics.append(f"  Agent {idx}: move_H={move_entropy:.3f}, attack_prob[punch]={attackP[-1]:.3f}")
        
        if is_collapsed:
            diagnostics.append(f"    ^^ DEGENERATE: collapsed to spin+punch")
    
    if prey_predicted_static:
        diagnostics.append(f"  Opponent model: predicts prey static (possible root cause)")
    
    return "\n".join(diagnostics)


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    env    = MalmoEnv(MISSION_XML)
    agent  = MARLeOM(lr=LR, alpha=ALPHA, gamma=GAMMA, tau=TAU, device=DEVICE,
                     levelMixWarmupSteps=WARMUP_STEPS,
                     levelMixRampSteps=WARMUP_STEPS * 2)
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    preyActor     = PreyActorNetwork().to("cpu")
    preyOptimizer = torch.optim.Adam(preyActor.parameters(), lr=LR)
    preyAlpha     = ALPHA

    ckptPath = os.path.join(CKPT_DIR, "marleom_latest.pt")
    if os.path.exists(ckptPath):
        agent.load(ckptPath)
        print(f"Auto-loaded checkpoint from {ckptPath}")

    predatorLogger = MARLLogger(
        algo_name="MARLeOM_predator",
        log_interval=LOG_INTERVAL,
        seed=42,
        log_file=os.path.join(CKPT_DIR, "marleom_predator_metrics.jsonl"),
    )
    preyLogger = MARLLogger(
        algo_name="MARLeOM_prey",
        log_interval=LOG_INTERVAL,
        seed=42,
        log_file=os.path.join(CKPT_DIR, "marleom_prey_metrics.jsonl"),
    )

    pred_eps, pred_wins = _restoreEpisodeCountersFromJsonl(predatorLogger.log_file)
    prey_eps, prey_wins = _restoreEpisodeCountersFromJsonl(preyLogger.log_file)
    predatorLogger._total_episodes = pred_eps
    predatorLogger._total_wins = pred_wins
    preyLogger._total_episodes = prey_eps
    preyLogger._total_wins = prey_wins
    if pred_eps > 0 or prey_eps > 0:
        print(f"Appended logs: predator episodes={pred_eps}, prey episodes={prey_eps}")

    totalSteps  = 0
    totalUpdates = 0
    forceWorldRestartPending = False
    winHistory  = []   # per-episode win flags (1 tag, 0 otherwise)

    print(f"{'Ep':>5} {'Steps':>6} {'Return':>8} {'Win%Cum':>8} {'Win%R100':>8} "
          f"{'CriticL':>8} {'ActorL':>8} {'OML0':>7} {'OML1':>7} "
          f"{'Alpha':>7} {'H_tgt':>6} {'H_pol':>6} {'EpSec':>6}")
    print("─" * 102)

    episode = 0
    try:
        while True:
            episode += 1
            t0             = time.time()
            obsAll         = env.reset(forceRestart=forceWorldRestartPending)
            if forceWorldRestartPending:
                print(f"[MARLeOM] Forced world restart at update {totalUpdates}")
                forceWorldRestartPending = False
            lastActionsAll = [NEUTRAL_ACTION] * NUM_AGENTS
            flatObs        = flattenObsAll(obsAll, lastActionsAll)

            episodeRewardPred = 0.0
            episodeRewardPrey = 0.0
            stepCount         = 0

            # Accumulators for per-episode loss averages
            criticLosses      = []
            actorLosses       = []
            omL0Losses        = []
            omL1Losses        = []
            alphaLosses       = []
            alphaVals         = []
            levelMixLast      = 0.0
            psiL0Last         = 1.0
            targetEntropyLast = float("nan")
            policyEntropyLast = float("nan")

            for step in range(MAX_STEPS):
                predObs = [obsAll[i] for i in PREDATOR_INDICES]

                if totalSteps < WARMUP_STEPS:
                    actions = [randomAction() for _ in range(NUM_AGENTS)]
                    actions[PREY_INDICES[0]] = preyPolicy(obsAll[PREY_INDICES[0]], predObs)
                else:
                    actions = agent.selectActions(flatObs, explore=True)
                    preyObsT    = torch.FloatTensor(flatObs[PREY_INDICES[0]]).unsqueeze(0)
                    pred0Oh     = torch.FloatTensor(flatObs[PREDATOR_INDICES[0], -ACTION_OH_DIM:]).unsqueeze(0)
                    preyActorIn = torch.cat([preyObsT, pred0Oh], dim=-1)
                    with torch.no_grad():
                        pm, pt, _, _ = preyActor.sampleAction(preyActorIn)
                        actions[PREY_INDICES[0]] = (pm.item(), pt.item(), 0)

                nextObsAll, rewardsAll, donesAll = env.step(actions)
                nextFlatObs = flattenObsAll(nextObsAll, actions)

                buffer.push(
                    obsAll     = flatObs,
                    actionsAll = np.array(actions,    dtype=np.float32),
                    rewardsAll = np.array(rewardsAll, dtype=np.float32),
                    nextObsAll = nextFlatObs,
                    dones      = np.array(donesAll,   dtype=np.float32),
                )

                flatObs    = nextFlatObs
                obsAll     = nextObsAll
                totalSteps += 1
                stepCount  += 1

                episodeRewardPred += sum(rewardsAll[i] for i in PREDATOR_INDICES)
                episodeRewardPrey += rewardsAll[PREY_INDICES[0]]

                if totalSteps >= WARMUP_STEPS and totalSteps % UPDATE_EVERY == 0:
                    if len(buffer) >= BATCH_SIZE:
                        losses = agent.update(buffer.sample(BATCH_SIZE))
                        totalUpdates += 1
                        criticLosses.append(losses["criticLoss"])
                        actorLosses.append(losses["actorLoss"])
                        omL0Losses.append(losses["om0Loss"])
                        omL1Losses.append(losses["om1Loss"])
                        alphaLosses.append(losses["alphaLoss"])
                        alphaVals.append(losses["alpha"])
                        levelMixLast      = losses["levelMix"]
                        psiL0Last         = losses["psiL0"]
                        targetEntropyLast = losses["targetEntropy"]
                        policyEntropyLast = losses["policyEntropy"]

                        if totalUpdates % WORLD_RESET_EVERY_UPDATES == 0:
                            forceWorldRestartPending = True

                        obsNp, actNp, rewNp, _, _ = buffer.sample(BATCH_SIZE)
                        preyIdx  = PREY_INDICES[0]
                        pred0Idx = PREDATOR_INDICES[0]
                        preyObsB  = torch.FloatTensor(obsNp[:, preyIdx, :])
                        pred0OhB  = torch.FloatTensor(obsNp[:, pred0Idx, -ACTION_OH_DIM:])
                        preyActIn = torch.cat([preyObsB, pred0OhB], dim=-1)
                        preyRewB  = torch.FloatTensor(rewNp[:, preyIdx])

                        pm, pt, logP, entropy = preyActor.sampleAction(preyActIn)
                        baseline = preyRewB.mean().detach()
                        preyLoss = -((preyRewB - baseline) * logP + preyAlpha * entropy).mean()
                        preyOptimizer.zero_grad()
                        preyLoss.backward()
                        preyOptimizer.step()

                if all(donesAll):
                    break

            # ---- Per-episode diagnostics (every 10 eps after warmup) ----
            tagged = env.preyWasTagged   # True if episode ended by tag
            winHistory.append(1 if tagged else 0)
            # Exact cumulative win rate across all episodes so far.
            cumulativeWinPct = 100.0 * sum(winHistory) / len(winHistory)

            # Smoother local trend for plotting/monitoring.
            rollingWindow = min(len(winHistory), 100)
            rollingWinPct = 100.0 * sum(winHistory[-rollingWindow:]) / rollingWindow

            epSec       = time.time() - t0
            avgCritic   = float(np.mean(criticLosses))  if criticLosses  else float("nan")
            avgActor    = float(np.mean(actorLosses))   if actorLosses   else float("nan")
            avgOmL0     = float(np.mean(omL0Losses))    if omL0Losses    else float("nan")
            avgOmL1     = float(np.mean(omL1Losses))    if omL1Losses    else float("nan")
            avgAlpha    = float(np.mean(alphaVals))      if alphaVals     else float("nan")

            diag      = diagnoseActionCollapse(flatObs, agent)
            collapsed = "SPIN" if "DEGENERATE" in diag else "    "

            print(f"{episode:5d} {totalSteps:6d} {episodeRewardPred:8.1f} {cumulativeWinPct:8.2f} {rollingWinPct:8.2f} "
                        f"{avgCritic:8.4f} {avgActor:8.4f} {avgOmL0:7.4f} {avgOmL1:7.4f} "
                        f"{avgAlpha:7.4f} {targetEntropyLast:6.3f} {policyEntropyLast:6.3f} {epSec:6.1f}  {collapsed}")

            predatorLogger.log_episode(
                episode        = predatorLogger._total_episodes + 1,
                episode_return = episodeRewardPred,
                win            = 1 if tagged else 0,
                prey_return    = episodeRewardPrey,
                survival_time  = stepCount,
            )
            preyLogger.log_episode(
                episode         = preyLogger._total_episodes + 1,
                episode_return  = episodeRewardPrey,
                win             = 0 if tagged else 1,
                predator_return = episodeRewardPred,
                survival_time   = stepCount,
            )

            if episode % LOG_INTERVAL == 0:
                predatorLogger.log_step(
                    step           = episode,
                    criticLoss     = avgCritic,
                    actorLoss      = avgActor,
                    omL0           = avgOmL0,
                    omL1           = avgOmL1,
                    alpha          = avgAlpha,
                    targetEntropy  = targetEntropyLast,
                    policyEntropy  = policyEntropyLast,
                    sec_per_ep     = epSec,
                )
                preyLogger.log_step(
                    step            = episode,
                    preyReturn      = episodeRewardPrey,
                    predatorReturn  = episodeRewardPred,
                    sec_per_ep      = epSec,
                )

            if episode % SAVE_EVERY == 0:
                path = os.path.join(CKPT_DIR, f"marleom_update{episode}.pt")
                latest_path = os.path.join(CKPT_DIR, "marleom_latest.pt")
                agent.save(path)
                agent.save(latest_path)
                print(f"  -> checkpoint saved {path}")

    except KeyboardInterrupt:
        print("\n[MARLeOM] Stopped by user (Ctrl+C). Saving final checkpoint...")

    agent.save(os.path.join(CKPT_DIR, "marleom_final.pt"))
    predatorLogger.print_final_summary()
    preyLogger.print_final_summary()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()