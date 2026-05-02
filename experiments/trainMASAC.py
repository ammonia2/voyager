"""
experiments/trainMASAC.py
=========================
Multi-Agent Soft Actor-Critic (MASAC) + Asynchronous Off-Policy PDC Training

PDC Architecture (Async Off-Policy Actor-Learner, Horgan et al. Ape-X 2018)
-----------------------------------------------------------------------------
  N worker processes  →  transitionQueue (mp.Queue)  →  GPU learner
       ↑                                                        |
       └──────────── actor weight broadcast (every K updates) ──┘

  Workers
  -------
  • Each worker owns ONE Malmo arena (portOffset = workerRank * NUM_AGENTS).
  • Workers hold CPU-only inference copies of both predator actors.
  • Each env step: push ONE joint transition dict to transitionQueue.
  • Every WEIGHT_SYNC_EVERY_STEPS steps: load fresh weights from weightQueue.

  Learner (main process)
  ----------------------
  • Owns the single PrioritizedReplayBuffer (local, no cross-process sharing).
  • Drains all transitionQueues into the buffer before each update.
  • Runs MASAC update on GPU, broadcasts weights back to workers.

NOTE: We use mp.Queue for transitions (not passing the buffer object) because
  with spawn method (Windows default), passed objects are pickled into each
  worker as private copies — the buffer would NOT be shared.

Run:
  python experiments/trainMASAC.py --numWorkers 2
  python experiments/trainMASAC.py --numWorkers 4
  python experiments/trainMASAC.py --resume
  python experiments/trainMASAC.py --evalOnly
"""
from __future__ import annotations
import os
import sys
import time
import random
import argparse
import json
import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.mappoEnv import MalmoEnv
from src.agents.masacAgent import MASACAgent
from src.utils.prioritizedReplay import PrioritizedReplayBuffer
from src.utils.obsUtils import (
    flattenObsAll, PREDATOR_INDICES, PREY_INDICES, NUM_AGENTS,
)
from src.utils.scriptedPolicies import get_pi_train_prey, get_pi_test_prey
from src.utils.logs import MARLLogger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MISSION_XML = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
CKPT_DIR    = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "masac")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BUFFER_CAPACITY       = 200_000  # ~715 MB for full buffer
PER_ALPHA             = 0.6      # priority exponent
PER_BETA              = 0.4      # importance sampling annealing start
PER_BETA_STEPS        = 100_000
LEARNING_STARTS       = 2_000    # joint transitions before learner starts

BATCH_SIZE            = 256
MAX_LEARNER_UPDATES   = 50_000
SAVE_EVERY_UPDATES    = 500
LOG_EVERY_UPDATES     = 2          # Reduced for testing (logs ~every 1 episode)
WEIGHT_BROADCAST_EVERY = 20      # push actor weights every N learner updates (low latency tolerance)

WEIGHT_SYNC_EVERY_STEPS  = 50    # worker: check for new weights every N steps
NEW_PREY_POLICY_EVERY_EPS = 20   # rotate scripted prey every N episodes

LR       = 3e-4
GAMMA    = 0.99
TAU      = 0.005        # polyak target update (conservative)
OM_COEFF = 0.5          # weight opponent modeling auxiliary loss
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ⚠️ QUEUE CONFIG FOR YOUR HARDWARE
# i5 12th gen + RTX 3060 + 16GB: Safe up to 6 workers
# With more workers, expect OOM (system memory ~2.5GB baseline)
TRANSITION_QUEUE_MAXSIZE = 500    # backpressure: worker blocks if learner is behind

PRED_NEUTRAL = (2, 0.0, 1)        # (move_stop, turn_zero, attack_off)
PREY_NEUTRAL = (2, 2, 0)          # (move_stop, turn_stop, unused)


def _neutralActionsAll() -> list:
    return [PRED_NEUTRAL if i in PREDATOR_INDICES else PREY_NEUTRAL
            for i in range(NUM_AGENTS)]


def _makeCPUActor():
    """Create an ActorNetwork in eval mode on CPU."""
    from src.models.masac.actorNetwork import ActorNetwork
    net = ActorNetwork()
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Utility: restore log counters from JSONL
# ---------------------------------------------------------------------------

def _restoreCounters(logPath: str) -> tuple:
    if not os.path.exists(logPath):
        return 0, 0
    eps, wins = 0, 0
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
            eps  += 1
            wins += int(rec.get("win", 0))
    return eps, wins


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def workerFn(
    workerRank:      int,
    numWorkers:      int,
    transitionQueue: mp.Queue,   # worker PUSHES joint transitions here
    weightQueue:     mp.Queue,   # worker READS new actor weights from here
    args:            argparse.Namespace,
):
    """
    Async worker: runs Malmo env, pushes ONE joint transition dict per step.

    Transition dict keys:
        obs0, obs1, preyObs            : (OBS_DIM,) float32
        act0, act1                     : (3,)       float32  [move_idx, turn, attack_idx]
        rew0, rew1                     : float
        nextObs0, nextObs1, nextPreyObs: (OBS_DIM,) float32
        done                           : float
        preyMove, preyTurn             : int   (ground-truth labels for OM loss)
    """
    torch.manual_seed(42 + workerRank)
    np.random.seed(42 + workerRank)
    random.seed(42 + workerRank)

    localActor0 = _makeCPUActor()
    localActor1 = _makeCPUActor()

    portOffset = workerRank * NUM_AGENTS
    env = MalmoEnv(MISSION_XML, portOffset=portOffset)

    preyPolicies  = get_pi_train_prey()
    preyPolicyIdx = workerRank % len(preyPolicies)
    preyPolicy    = preyPolicies[preyPolicyIdx]
    episodeCount  = 0

    lastActionsAll = _neutralActionsAll()
    obsAll         = env.reset()
    flatObsAll     = flattenObsAll(obsAll, lastActionsAll)

    stepsSinceSync = 0
    print(f"[Worker {workerRank}] Started. portOffset={portOffset}, "
          f"prey={preyPolicy.name}")

    while True:
        # ── Sync actor weights from learner ──────────────────────────────
        if stepsSinceSync >= WEIGHT_SYNC_EVERY_STEPS:
            latestWeights = None
            while not weightQueue.empty():
                try:
                    msg = weightQueue.get_nowait()
                    if msg == "STOP":
                        print(f"[Worker {workerRank}] Stop received. Exiting.")
                        return
                    latestWeights = msg
                except Exception:
                    break
            if latestWeights is not None:
                localActor0.load_state_dict(latestWeights["actor0"])
                localActor1.load_state_dict(latestWeights["actor1"])
            stepsSinceSync = 0

        # ── Predator actions (stochastic inference, no grad) ─────────────
        obs0T = torch.FloatTensor(flatObsAll[PREDATOR_INDICES[0]]).unsqueeze(0)
        obs1T = torch.FloatTensor(flatObsAll[PREDATOR_INDICES[1]]).unsqueeze(0)
        action0 = localActor0.stochasticAction(obs0T)   # (move, turn, attack)
        action1 = localActor1.stochasticAction(obs1T)

        # ── Prey action (fixed scripted policy) ──────────────────────────
        preyObsIdx           = PREY_INDICES[0]
        preyMove, preyTurn   = preyPolicy(obsAll[preyObsIdx], preyObsIdx)
        preyActionTuple      = (preyMove, preyTurn, 0)

        # ── Assemble action list and step env ────────────────────────────
        actionsAll = list(lastActionsAll)
        actionsAll[PREDATOR_INDICES[0]] = action0
        actionsAll[PREDATOR_INDICES[1]] = action1
        actionsAll[PREY_INDICES[0]]     = preyActionTuple

        nextObsAll, rewardsAll, donesAll = env.step(actionsAll)
        nextFlatObsAll = flattenObsAll(nextObsAll, actionsAll)
        done = all(donesAll)

        # ── Push ONE joint transition to learner ──────────────────────────
        transitionQueue.put({
            "obs0":        flatObsAll[PREDATOR_INDICES[0]].copy(),
            "obs1":        flatObsAll[PREDATOR_INDICES[1]].copy(),
            "preyObs":     flatObsAll[PREY_INDICES[0]].copy(),
            "act0":        np.array(action0, dtype=np.float32),
            "act1":        np.array(action1, dtype=np.float32),
            "rew0":        float(rewardsAll[PREDATOR_INDICES[0]]),
            "rew1":        float(rewardsAll[PREDATOR_INDICES[1]]),
            "nextObs0":    nextFlatObsAll[PREDATOR_INDICES[0]].copy(),
            "nextObs1":    nextFlatObsAll[PREDATOR_INDICES[1]].copy(),
            "nextPreyObs": nextFlatObsAll[PREY_INDICES[0]].copy(),
            "done":        float(done),
            "preyMove":    int(preyMove),
            "preyTurn":    int(preyTurn),
        })

        stepsSinceSync += 1
        flatObsAll      = nextFlatObsAll
        lastActionsAll  = actionsAll
        obsAll          = nextObsAll

        if done:
            episodeCount += 1
            if episodeCount % NEW_PREY_POLICY_EVERY_EPS == 0:
                preyPolicyIdx = (preyPolicyIdx + 1) % len(preyPolicies)
                preyPolicy    = preyPolicies[preyPolicyIdx]
            obsAll         = env.reset()
            lastActionsAll = _neutralActionsAll()
            flatObsAll     = flattenObsAll(obsAll, lastActionsAll)


# ---------------------------------------------------------------------------
# Learner helpers
# ---------------------------------------------------------------------------

def _drainQueues(
    transitionQueues: list,
    buffer: PrioritizedReplayBuffer,
    maxPerQueue: int = 200,
):
    """Drain worker transition queues into the local replay buffer."""
    for q in transitionQueues:
        drained = 0
        while drained < maxPerQueue:
            try:
                t = q.get_nowait()
                buffer.push(
                    obs0        = t["obs0"],
                    obs1        = t["obs1"],
                    preyObs     = t["preyObs"],
                    act0        = t["act0"],
                    act1        = t["act1"],
                    rew0        = t["rew0"],
                    rew1        = t["rew1"],
                    nextObs0    = t["nextObs0"],
                    nextObs1    = t["nextObs1"],
                    nextPreyObs = t["nextPreyObs"],
                    done        = t["done"],
                    preyMove    = t["preyMove"],
                    preyTurn    = t["preyTurn"],
                )
                drained += 1
            except Exception:
                break


# ---------------------------------------------------------------------------
# Learner process (runs in main process on GPU)
# ---------------------------------------------------------------------------

def learnerFn(
    transitionQueues: list,   # one mp.Queue per worker (learner reads)
    weightQueues:     list,   # one mp.Queue per worker (learner writes)
    args:             argparse.Namespace,
):
    """
    GPU learner: owns the replay buffer, drains worker queues, runs MASAC updates.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ⚠️ HARDWARE CHECK
    print(f"[Learner] GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Learner] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Learner] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    numWorkers = len(transitionQueues)
    estimatedMemoryGB = 2.5 + (0.2 * numWorkers)
    print(f"[Learner] Estimated system memory: {estimatedMemoryGB:.1f} GB (for {numWorkers} workers)")
    if numWorkers > 6:
        print(f"[Learner] ⚠️  WARNING: {numWorkers} workers may exceed 16GB RAM. Consider reducing to 4-6.")

    # ── Local buffer (not shared — lives only in this process) ────────────
    buffer = PrioritizedReplayBuffer(
        capacity  = BUFFER_CAPACITY,
        alpha     = PER_ALPHA,
        beta      = PER_BETA,
        betaSteps = PER_BETA_STEPS,
    )

    agent = MASACAgent(device=DEVICE, lr=LR, gamma=GAMMA, tau=TAU, omCoeff=OM_COEFF)

    ckptPath = os.path.join(CKPT_DIR, "masac_latest.pt")
    if args.resume and os.path.exists(ckptPath):
        agent.load(ckptPath)
        print(f"[Learner] Resumed from {ckptPath}")

    # ── Broadcast initial weights to workers ──────────────────────────────
    initWeights = agent.getActorState()
    for q in weightQueues:
        q.put(initWeights)

    # ── Logger ────────────────────────────────────────────────────────────
    logFile = os.path.join(CKPT_DIR, "masac_metrics.jsonl")
    logger  = MARLLogger(algo_name="MASAC", log_interval=LOG_EVERY_UPDATES, seed=0,
                         log_file=logFile)
    eps, wins = _restoreCounters(logFile)
    logger._total_episodes = eps
    logger._total_wins     = wins

    # ── Wait until buffer has enough transitions ──────────────────────────
    print(f"[Learner] Collecting {LEARNING_STARTS} transitions before training...")
    while len(buffer) < LEARNING_STARTS:
        _drainQueues(transitionQueues, buffer)
        time.sleep(0.5)
    print(f"[Learner] Buffer ready ({len(buffer)} transitions). Starting updates.")

    # ── Main learner loop ─────────────────────────────────────────────────
    for updateNum in range(1, MAX_LEARNER_UPDATES + 1):
        t0 = time.time()

        # Drain fresh transitions from workers
        _drainQueues(transitionQueues, buffer)

        # Sample batch from PER
        batch = buffer.sample(BATCH_SIZE)

        # MASAC update with correct per-predator inputs
        losses = agent.update(
            obs0         = batch["obs0"],
            obs1         = batch["obs1"],
            preyObs      = batch["preyObs"],
            act0         = batch["act0"],
            act1         = batch["act1"],
            rew0         = batch["rew0"],
            rew1         = batch["rew1"],
            nextObs0     = batch["nextObs0"],
            nextObs1     = batch["nextObs1"],
            nextPreyObs  = batch["nextPreyObs"],
            dones        = batch["dones"],
            isWeights    = batch["weights"],
            preyMoveTrue = batch["preyMove"],
            preyTurnTrue = batch["preyTurn"],
        )

        # Update PER priorities
        buffer.updatePriorities(batch["dataIndices"], losses["tdErrors"])

        # Broadcast updated weights to workers
        if updateNum % WEIGHT_BROADCAST_EVERY == 0:
            weightState = agent.getActorState()
            for q in weightQueues:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Exception:
                        break
                q.put(weightState)

        # Logging
        if updateNum % LOG_EVERY_UPDATES == 0:
            elapsed = time.time() - t0
            logger.log_step(
                step           = updateNum,
                criticLoss     = losses["criticLoss"],
                actorLoss0     = losses["actorLoss0"],
                actorLoss1     = losses["actorLoss1"],
                omLoss0        = losses["omLoss0"],
                omLoss1        = losses["omLoss1"],
                alpha0         = losses["alpha0"],
                alpha1         = losses["alpha1"],
                bufferSize     = len(buffer),
                sec_per_update = elapsed,
            )

        # Checkpointing
        if updateNum % SAVE_EVERY_UPDATES == 0:
            agent.save(os.path.join(CKPT_DIR, f"masac_update{updateNum}.pt"))
            agent.save(ckptPath)

    # Stop workers
    for q in weightQueues:
        q.put("STOP")

    agent.save(os.path.join(CKPT_DIR, "masac_final.pt"))
    logger.print_final_summary()
    print("[Learner] Training complete.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evalMASAC(numEvalEps: int = 50, seenPolicies: bool = True, unseenPolicies: bool = True):
    """Evaluate trained MASAC predators against seen and unseen prey policies."""
    ckptPath = os.path.join(CKPT_DIR, "masac_final.pt")
    if not os.path.exists(ckptPath):
        ckptPath = os.path.join(CKPT_DIR, "masac_latest.pt")
    assert os.path.exists(ckptPath), f"No checkpoint found at {ckptPath}"

    agent = MASACAgent(device=DEVICE)
    agent.load(ckptPath)
    agent.actor0.eval()
    agent.actor1.eval()

    env    = MalmoEnv(MISSION_XML, portOffset=0)
    logger = MARLLogger(algo_name="MASAC_eval", log_interval=numEvalEps, seed=0)

    policyGroups = []
    if seenPolicies:
        policyGroups.append(("seen",   get_pi_train_prey()))
    if unseenPolicies:
        policyGroups.append(("unseen", get_pi_test_prey()))

    for oppType, policies in policyGroups:
        for ep in range(numEvalEps):
            preyPolicy     = policies[ep % len(policies)]
            lastActionsAll = _neutralActionsAll()
            obsAll         = env.reset()
            flatObsAll     = flattenObsAll(obsAll, lastActionsAll)
            epReturn       = 0.0
            omCorrectMove  = []
            omCorrectTurn  = []
            done = False

            while not done:
                obs0T = torch.FloatTensor(flatObsAll[PREDATOR_INDICES[0]]).unsqueeze(0)
                obs1T = torch.FloatTensor(flatObsAll[PREDATOR_INDICES[1]]).unsqueeze(0)
                action0 = agent.actor0.deterministicAction(obs0T)
                action1 = agent.actor1.deterministicAction(obs1T)

                preyObsIdx         = PREY_INDICES[0]
                preyMove, preyTurn = preyPolicy(obsAll[preyObsIdx], preyObsIdx)
                preyAction         = (preyMove, preyTurn, 0)

                actionsAll = list(lastActionsAll)
                actionsAll[PREDATOR_INDICES[0]] = action0
                actionsAll[PREDATOR_INDICES[1]] = action1
                actionsAll[PREY_INDICES[0]]     = preyAction

                with torch.no_grad():
                    enc0 = agent.actor0.encoder(obs0T)
                    movePred, turnPred = agent.actor0.omHead.predictProbs(enc0)
                    omCorrectMove.append(int(movePred.argmax().item() == preyMove))
                    omCorrectTurn.append(int(turnPred.argmax().item() == preyTurn))

                nextObsAll, rewardsAll, donesAll = env.step(actionsAll)
                epReturn      += sum(rewardsAll[i] for i in PREDATOR_INDICES) / len(PREDATOR_INDICES)
                done           = all(donesAll)
                flatObsAll     = flattenObsAll(nextObsAll, actionsAll)
                lastActionsAll = actionsAll
                obsAll         = nextObsAll

            winFlag = 1 if env.preyWasTagged else 0
            oppAcc  = (np.mean(omCorrectMove) + np.mean(omCorrectTurn)) / 2.0

            logger.log_eval(
                episode        = ep + 1,
                episode_return = epReturn,
                win            = winFlag,
                opponent_acc   = oppAcc,
                opponent_type  = oppType,
            )

    logger.print_eval_summary(
        episode           = numEvalEps,
        n_seen_episodes   = numEvalEps if seenPolicies   else 0,
        n_unseen_episodes = numEvalEps if unseenPolicies else 0,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parseArgs() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MASAC + Async PDC")
    p.add_argument("--numWorkers", type=int, default=2,
                   help="Async environment workers (recommended: 2 for 16GB RAM)")
    p.add_argument("--resume",   action="store_true",
                   help="Resume from masac_latest.pt")
    p.add_argument("--evalOnly", action="store_true",
                   help="Skip training; run evaluation from masac_final.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = parseArgs()

    if args.evalOnly:
        evalMASAC(numEvalEps=50, seenPolicies=True, unseenPolicies=True)
        sys.exit(0)

    mp.set_start_method("spawn", force=True)

    # One queue per worker: workers PUSH transitions, learner READS
    transitionQueues = [
        mp.Queue(maxsize=TRANSITION_QUEUE_MAXSIZE)
        for _ in range(args.numWorkers)
    ]
    # One queue per worker: learner PUSHES weights, workers READ
    weightQueues = [mp.Queue(maxsize=2) for _ in range(args.numWorkers)]

    workerProcs = []
    for rank in range(args.numWorkers):
        p = mp.Process(
            target = workerFn,
            args   = (rank, args.numWorkers,
                      transitionQueues[rank], weightQueues[rank], args),
            daemon = True,
        )
        p.start()
        workerProcs.append(p)
        print(f"[Main] Worker {rank} spawned (PID {p.pid})")

    try:
        learnerFn(transitionQueues, weightQueues, args)
    finally:
        for p in workerProcs:
            p.terminate()
        for p in workerProcs:
            p.join(timeout=5)
        print("[Main] All workers terminated.")
