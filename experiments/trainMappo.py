"""
MAPPO + Voxel OM Encoder | PDC data-parallel distributed SGD (Liu et al. 2025)

Action-space summary:
  Predator: move (discrete 3) | turn (CONTINUOUS [-1,1]) | attack (discrete 2)
  Prey:     move (discrete 3) | turn (discrete 3)         | NO attack

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
import tempfile
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributions import Categorical, Normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.mappoEnv import MalmoEnv, MOVE_CMDS, PREY_TURN_CMDS, ATTACK_CMDS
from src.agents.mappo import MAPPO
from src.utils.rolloutBuffer import RolloutBuffer
from src.utils.logs import MARLLogger
from src.utils.obsUtils import (
    flattenObsAll, buildGlobalState,
    PREDATOR_INDICES, PREY_INDICES, NUM_AGENTS,
)

MISSION_XML = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
CKPT_DIR    = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

# Hyperparameters
ROLLOUT_STEPS  = 512
PPO_EPOCHS     = 4
MINI_BATCH     = 128
GAMMA          = 0.99
LAMDA          = 0.95
CLIP_EPS       = 0.15
ENTROPY_COEFF  = 0.01
VAL_COEFF      = 0.5
OM_COEFF       = 0.5
LR             = 3e-4
MAX_UPDATES    = 5000
SAVE_EVERY     = 10
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# Neutral (do-nothing) actions — differ by agent type
#   Predator: (move_stop=2, turn_zero=0.0, attack_off=1)
#   Prey:     (move_stop=2, turn_stop=2,   unused=0)
PRED_NEUTRAL = (2, 0.0, 1)
PREY_NEUTRAL = (2, 2, 0)

def _neutralActionsAll() -> list[tuple]:
    return [PRED_NEUTRAL if i in PREDATOR_INDICES else PREY_NEUTRAL
            for i in range(NUM_AGENTS)]

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
    epPredReward: float,
    preyRewardEma: float,
) -> tuple[list[dict], list[tuple], float, float, float, list[tuple], float]:
    """
    Fills buffer with ROLLOUT_STEPS transitions.
    Returns:
      obsAll, lastActionsAll, lastValue, preyLastValue, 
      ongoing_epPredReward, finished_eps, updated preyRewardEma
    """
    buffer.reset()
    flatObs = flattenObsAll(obsAll, lastActionsAll)

    finished_eps = []
    done         = False

    for _ in range(buffer.T):
        globalState = buildGlobalState(flatObs)

        # ---------- Predator actions + log-probs (continuous turn) ----------
        actions     = agent.selectActions(flatObs, explore=True)
        logProbsArr = np.zeros(len(PREDATOR_INDICES), dtype=np.float32)

        with torch.no_grad():
            obsT = torch.FloatTensor(flatObs).to(agent.device)
            for localIdx, agentIdx in enumerate(PREDATOR_INDICES):
                moveP, turn_mean, turn_std, attackP = agent.actor(obsT[agentIdx].unsqueeze(0))
                m = torch.tensor([int(actions[agentIdx][0])])
                t = torch.tensor([[float(actions[agentIdx][1])]])   # (1,1) for Normal
                a = torch.tensor([int(actions[agentIdx][2])])
                lp = (
                    Categorical(moveP).log_prob(m)
                    + Normal(turn_mean, turn_std).log_prob(t).sum(-1)
                    + Categorical(attackP).log_prob(a)
                )
                logProbsArr[localIdx] = lp.item()

        # Centralised critic value
        value = agent.getValue(globalState)

        # ---------- Prey action + log-prob (discrete move+turn, no attack) ----------
        preyIdx    = PREY_INDICES[0]
        preyAction = agent.selectPreyAction(flatObs[preyIdx], explore=True)
        actions[preyIdx] = preyAction

        with torch.no_grad():
            preyObsT = torch.FloatTensor(flatObs[preyIdx]).unsqueeze(0).to(agent.device)
            mpP, tpP = agent.preyActor(preyObsT)
            pm = torch.tensor([preyAction[0]])
            pt = torch.tensor([preyAction[1]])
            preyLogProb = (
                Categorical(mpP).log_prob(pm)
                + Categorical(tpP).log_prob(pt)
            ).item()

        # Prey value = EMA of prey reward (running mean baseline for REINFORCE)
        preyValue = preyRewardEma

        nextObsAll, rewardsAll, donesAll = env.step(actions)
        nextFlatObs = flattenObsAll(nextObsAll, actions)
        done        = all(donesAll)

        buffer.add(
            flatObs, globalState,
            np.array(actions,    dtype=np.float32),  # float32: turn is continuous
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
            finished_eps.append((epPredReward, winFlag))
            epPredReward = 0.0
            obsAll  = env.reset()
            flatObs = flattenObsAll(obsAll, _neutralActionsAll())

    # Bootstrap values for GAE
    lastValue     = agent.getValue(buildGlobalState(flatObs)) if not done else 0.0
    preyLastValue = preyRewardEma if not done else 0.0

    return obsAll, lastActionsAll, lastValue, preyLastValue, epPredReward, finished_eps, preyRewardEma


# ------------------------------------------------------------------
# Worker process
# ------------------------------------------------------------------

def workerFn(rank: int, worldSize: int, args: argparse.Namespace):
    if worldSize > 1:
        # File-based rendezvous: no TCP sockets used for coordination.
        # Avoids the Windows gloo bug (WSAEADDRNOTAVAIL / error 10049) where
        # gloo tries to bind its local socket to the machine hostname instead
        # of loopback regardless of MASTER_ADDR.
        rendUri = "file:///" + args.rendFile.replace("\\", "/")
        dist.init_process_group(
            backend="gloo",
            init_method=rendUri,
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

    if rank == 0:
        ckptPath = os.path.join(CKPT_DIR, "mappo_latest.pt")
        if os.path.exists(ckptPath):
            agent.load(ckptPath)
            print(f"[rank {rank}] Auto-loaded checkpoint from {ckptPath}")

    # Broadcast rank-0 weights to all workers before training starts
    if worldSize > 1:
        _broadcastParams(list(agent.allParams()), src=0)
        dist.barrier()

    lastActionsAll = _neutralActionsAll()
    obsAll         = env.reset()
    epPredReward   = 0.0
    preyRewardEma  = 0.0
    totalSteps     = 0

    logger = None
    if rank == 0:
        log_file = os.path.join(CKPT_DIR, "mappo_training.log")
        logger = MARLLogger(algo_name="MAPPO", log_interval=10, seed=42, log_file=log_file)

    for updateNum in range(1, MAX_UPDATES + 1):
        t0 = time.time()

        (obsAll, lastActionsAll, lastValue, preyLastValue,
         epPredReward, finished_eps, preyRewardEma) = collectRollout(
            env, agent, buffer, obsAll, lastActionsAll,
            epPredReward, preyRewardEma,
        )

        totalSteps += ROLLOUT_STEPS * worldSize

        rollout = buffer.get(lastValue, preyLastValue)

        def reduceFn(params):
            if worldSize > 1:
                _allReduceGrads(params, worldSize)
                dist.barrier()

        losses = agent.update(rollout, reduceFn)

        if rank == 0 and logger is not None:
            for rw, wf in finished_eps:
                logger._total_episodes += 1  # Track local episodes manually to trigger summary on local 0
                logger.log_episode(
                    episode        = logger._total_episodes,
                    episode_return = rw,
                    win            = wf,
                    opponent_type  = "seen"
                )

            # Log loss information every 10 updates
            if updateNum % 10 == 0:
                elapsed = time.time() - t0
                logger.log_step(
                    step        = totalSteps,
                    update      = updateNum,
                    policyLoss  = losses["policyLoss"],
                    valueLoss   = losses["valueLoss"],
                    entropy     = losses["entropy"],
                    omLoss      = losses["omLoss"],
                    sec_per_up  = elapsed
                )

        # Synchronize episode count across workers and terminate if reached
        epT = torch.tensor([logger._total_episodes if rank == 0 and logger else 0], dtype=torch.long, device=DEVICE)
        if worldSize > 1:
            dist.broadcast(epT, src=0)
        
        if epT.item() >= 5000:
            if rank == 0:
                print("\n[rank 0] Reached 5000 episodes limit. Stopping training.")
            break

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

    if worldSize > 1:
        # Compute & clean up the rendezvous file BEFORE spawning workers.
        # File-based rendezvous is the only reliable approach on Windows
        # because gloo's TCP rendezvous resolves the machine hostname for
        # binding regardless of MASTER_ADDR (WSAEADDRNOTAVAIL / error 10049).
        rendFile = os.path.join(
            tempfile.gettempdir(),
            f"mappo_dist_{args.masterPort}.rend",
        )
        if os.path.exists(rendFile):
            os.remove(rendFile)
        args.rendFile = rendFile
    else:
        args.rendFile = ""

    if worldSize == 1:
        workerFn(0, 1, args)
    else:
        mp.spawn(workerFn, args=(worldSize, args), nprocs=worldSize, join=True)