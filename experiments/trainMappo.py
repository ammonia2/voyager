"""
MAPPO + Voxel OM Encoder | PDC data-parallel distributed SGD (Liu et al. 2025)

Workers:
  - Each worker holds a full model replica + its own Malmo arena
  - Rollouts collected in parallel with no inter-worker communication
  - After collection: gradients averaged via dist.all_reduce (gloo backend)
  - All replicas apply the identical averaged gradient -> stay synchronised
  - No replay buffer, no staleness

Run:
  python trainMAPPO.py --numWorkers 1   # single worker (baseline)
  python trainMAPPO.py --numWorkers 4   # PDC with 4 workers
"""
from __future__ import annotations
import os
import sys
import time
import math
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.mappoEnv import MalmoEnv, MOVE_CMDS, TURN_CMDS, ATTACK_CMDS
from src.agents.mappo import MAPPO
from src.utils.rolloutBuffer import RolloutBuffer
from src.utils.obsUtils import (
    flattenObsAll, buildGlobalState,
    PREDATOR_INDICES, PREY_INDICES, NUM_AGENTS,
)

MISSION_XML = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
CKPT_DIR    = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

# Hyperparameters
ROLLOUT_STEPS  = 512
PPO_EPOCHS     = 1
MINI_BATCH     = 512
GAMMA          = 0.99
LAMDA          = 0.95
CLIP_EPS       = 0.15
ENTROPY_COEFF  = 0.01
VAL_COEFF      = 0.5
OM_COEFF       = 0.5
LR             = 3e-4
MAX_UPDATES    = 5000
SAVE_EVERY     = 100
NEUTRAL_ACTION = (2, 2, 1)
DEVICE         = "cpu"

# ------------------------------------------------------------------
# Distributed helpers
# ------------------------------------------------------------------

def _allReduceGrads(params, worldSize: int):
    """Sum-reduce then divide: equivalent to averaging gradients."""
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(worldSize)


def _broadcastParams(params, src: int = 0):
    """Broadcast initial weights from rank 0 to all workers."""
    for p in params:
        dist.broadcast(p.data, src=src)


# ------------------------------------------------------------------
# Rollout collection
# ------------------------------------------------------------------

def collectRollout(
    env: MalmoEnv,
    agent: MAPPO,
    buffer: RolloutBuffer,
    obsAll: list[dict],
    lastActionsAll: list[tuple],
    episodeNum: int,
    preyRewardEma: float,
) -> tuple[list[dict], list[tuple], float, float, float, float, float]:
    """
    Fills buffer with ROLLOUT_STEPS transitions.
    Returns:
      obsAll, lastActionsAll, lastValue, preyLastValue,
      epPredReward, winFlag, updated preyRewardEma
    """
    buffer.reset()
    flatObs = flattenObsAll(obsAll, lastActionsAll)

    epPredReward = 0.0
    winFlag      = 0.0
    done         = False

    for _ in range(buffer.T):
        globalState = buildGlobalState(flatObs)

        # Predator actions + log probs
        actions     = agent.selectActions(flatObs, explore=True)
        logProbsArr = np.zeros(len(PREDATOR_INDICES), dtype=np.float32)

        with torch.no_grad():
            obsT = torch.FloatTensor(flatObs).to(agent.device)
            for localIdx, agentIdx in enumerate(PREDATOR_INDICES):
                moveP, turnP, attackP = agent.actor(obsT[agentIdx].unsqueeze(0))
                m = torch.tensor([actions[agentIdx][0]])
                t = torch.tensor([actions[agentIdx][1]])
                a = torch.tensor([actions[agentIdx][2]])
                lp = (
                    torch.distributions.Categorical(moveP).log_prob(m)
                    + torch.distributions.Categorical(turnP).log_prob(t)
                    + torch.distributions.Categorical(attackP).log_prob(a)
                )
                logProbsArr[localIdx] = lp.item()

        # Centralised critic value
        value = agent.getValue(globalState)

        # Prey action
        preyIdx     = PREY_INDICES[0]
        preyAction = agent.selectPreyAction(flatObs[preyIdx], explore=True)
        actions[preyIdx] = preyAction

        # Prey log prob
        with torch.no_grad():
            preyObsT   = torch.FloatTensor(flatObs[preyIdx]).unsqueeze(0).to(agent.device)
            mpP, tpP, apP = agent.preyActor(preyObsT)
            pm = torch.tensor([preyAction[0]])
            pt = torch.tensor([preyAction[1]])
            pa = torch.tensor([preyAction[2]])
            preyLogProb = (
                torch.distributions.Categorical(mpP).log_prob(pm)
                + torch.distributions.Categorical(tpP).log_prob(pt)
                + torch.distributions.Categorical(apP).log_prob(pa)
            ).item()

        # Prey value = EMA of prey reward (running mean baseline for REINFORCE)
        preyValue = preyRewardEma

        nextObsAll, rewardsAll, donesAll = env.step(actions)
        nextFlatObs = flattenObsAll(nextObsAll, actions)
        done        = all(donesAll)

        buffer.add(
            flatObs, globalState,
            np.array(actions,    dtype=np.int64),
            np.array(rewardsAll, dtype=np.float32),
            done,
            logProbsArr, value, preyLogProb, preyValue,
        )

        # Update EMA prey reward baseline
        preyRewardEma = 0.99 * preyRewardEma + 0.01 * rewardsAll[preyIdx]

        epPredReward   += sum(rewardsAll[i] for i in PREDATOR_INDICES) / len(PREDATOR_INDICES)
        lastActionsAll  = list(actions)
        flatObs         = nextFlatObs
        obsAll          = nextObsAll

        if done:
            winFlag = 1.0 if env.preyWasTagged else 0.0
            obsAll  = env.reset()
            flatObs = flattenObsAll(obsAll, [NEUTRAL_ACTION] * NUM_AGENTS)

    # Bootstrap values for GAE
    lastValue     = agent.getValue(buildGlobalState(flatObs)) if not done else 0.0
    preyLastValue = preyRewardEma if not done else 0.0

    return obsAll, lastActionsAll, lastValue, preyLastValue, epPredReward, winFlag, preyRewardEma


# ------------------------------------------------------------------
# Worker process
# ------------------------------------------------------------------

def workerFn(rank: int, worldSize: int, args: argparse.Namespace):
    if worldSize > 1:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{args.masterPort}",
            world_size=worldSize,
            rank=rank,
        )

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)

    os.makedirs(CKPT_DIR, exist_ok=True)

    agent  = MAPPO(
        lr=LR, gamma=GAMMA, lamda=LAMDA, clipEps=CLIP_EPS,
        entropyCoeff=ENTROPY_COEFF, valCoeff=VAL_COEFF, omCoeff=OM_COEFF,
        ppoEpochs=PPO_EPOCHS, miniBatchSize=MINI_BATCH, device=DEVICE,
    )
    env    = MalmoEnv(MISSION_XML, portOffset=rank * NUM_AGENTS)
    buffer = RolloutBuffer(ROLLOUT_STEPS)

    if args.resume and rank == 0:
        ckptPath = os.path.join(CKPT_DIR, "mappo_latest.pt")
        if os.path.exists(ckptPath):
            agent.load(ckptPath)
            print(f"[rank {rank}] Loaded {ckptPath}")

    # Broadcast rank-0 weights to all workers before training starts
    if worldSize > 1:
        _broadcastParams(list(agent.allParams()), src=0)
        dist.barrier()

    lastActionsAll = [NEUTRAL_ACTION] * NUM_AGENTS
    obsAll         = env.reset()
    episodeNum     = 0
    preyRewardEma  = 0.0
    winHistory: list[float] = []
    totalSteps = 0

    if rank == 0:
        header = (
            f"{'Update':>7} {'TotalSteps':>11} {'Ep':>6} "
            f"{'PredRew':>8} {'Win%':>7} "
            f"{'PolLoss':>9} {'ValLoss':>9} {'Entropy':>8} {'OMLoss':>8} "
            f"{'Sec/Up':>7}"
        )
        print(header)
        print("-" * len(header))

    for updateNum in range(1, MAX_UPDATES + 1):
        t0 = time.time()

        (obsAll, lastActionsAll, lastValue, preyLastValue,
         epPredReward, winFlag, preyRewardEma) = collectRollout(
            env, agent, buffer, obsAll, lastActionsAll,
            episodeNum, preyRewardEma,
        )

        episodeNum += 1
        winHistory.append(winFlag)
        totalSteps += ROLLOUT_STEPS * worldSize  # each worker collected ROLLOUT_STEPS

        rollout = buffer.get(lastValue, preyLastValue)

        # Compute gradients locally (no optimiser step yet)
        losses = agent.computeGradients(rollout)
        agent.computePreyGradients(rollout)

        # All-reduce: average gradients across workers
        if worldSize > 1:
            _allReduceGrads(list(agent.allParams()), worldSize)
            dist.barrier()

        # Apply averaged gradients
        agent.applyGradients()

        if rank == 0 and updateNum % 10 == 0:
            winRate = 100.0 * sum(winHistory[-100:]) / max(1, min(len(winHistory), 100))
            elapsed = time.time() - t0
            print(
                f"{updateNum:7d} {totalSteps:11d} {episodeNum:6d} "
                f"{epPredReward:8.2f} {winRate:7.2f} "
                f"{losses['policyLoss']:9.4f} {losses['valueLoss']:9.4f} "
                f"{losses['entropy']:8.4f} {losses['omLoss']:8.4f} "
                f"{elapsed:7.1f}"
            )

        if rank == 0 and updateNum % SAVE_EVERY == 0:
            path = os.path.join(CKPT_DIR, f"mappo_update{updateNum}.pt")
            agent.save(path)
            agent.save(os.path.join(CKPT_DIR, "mappo_latest.pt"))
            print(f"  -> checkpoint saved {path}")

    if rank == 0:
        agent.save(os.path.join(CKPT_DIR, "mappo_final.pt"))
        print("Training complete.")

    if worldSize > 1:
        dist.destroy_process_group()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def parseArgs() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MAPPO with voxel OM encoder (PDC)")
    p.add_argument("--numWorkers", type=int, default=1,     help="N workers: 1, 2, 4, or 6")
    p.add_argument("--masterPort", type=int, default=29500, help="gloo rendezvous port")
    p.add_argument("--resume",     action="store_true",     help="resume from mappo_latest.pt")
    return p.parse_args()


if __name__ == "__main__":
    args      = parseArgs()
    worldSize = args.numWorkers

    if worldSize == 1:
        workerFn(0, 1, args)
    else:
        mp.spawn(workerFn, args=(worldSize, args), nprocs=worldSize, join=True)