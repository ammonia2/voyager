"""
trainBR.py
==========
Phase 1 of OMIS pipeline — distributed PDC version.

Each worker runs its own Malmo arena and collects rollouts independently.
After each update, gradients are averaged across all workers (all_reduce),
keeping all replicas synchronized — same pattern as trainMappo.py.

Run:
  python experiments/trainBR.py --policy 0 --numWorkers 1   # single
  python experiments/trainBR.py --policy 0 --numWorkers 2   # 2 workers
"""

import os
import sys
import time
import tempfile
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.malmoEnvOmis import MalmoEnv
from src.agents.brAgent import PPOTrainer, TOTAL_EPISODES
from src.utils.obsUtils import flattenObs
from src.utils.scriptedPolicies import (
    get_prey_policy_by_index,
    decode_pred_action_30,
    turn_bin5_to_cont,
)
from src.utils.logs import MARLLogger


MISSION_XML = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
CKPT_DIR    = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "br")

LOG_INTERVAL  = 10
SAVE_INTERVAL = 50

SELF_AGENT_IDX   = 0
TEAMMATE_IDX     = 1
PREY_IDX         = 2
PORTS_PER_WORKER = 3


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def encode_obs(agent_idx, obs_all, last_actions):
    """Flatten one agent's observation to a numpy array (147-dim)."""
    return flattenObs(agent_idx, obs_all[agent_idx], obs_all, last_actions)


def _allReduceGrads(params, worldSize):
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(worldSize)


# ─────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────

def workerFn(rank, worldSize, args):
    # ── Distributed setup ──────────────────────────────────────
    if worldSize > 1:
        rendUrl = f"file:///{args.rendFile.replace(chr(92), '/')}"
        dist.init_process_group(
            backend    = "gloo",
            init_method= rendUrl,
            rank       = rank,
            world_size = worldSize,
        )

    device = torch.device("cpu")

    # Each worker gets its own 3-port Malmo arena
    env = MalmoEnv(MISSION_XML, portOffset=rank * PORTS_PER_WORKER)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"br_policy_{args.policy}.pt")

    prey_policy = get_prey_policy_by_index(args.policy)
    trainer     = PPOTrainer(policy_idx=args.policy, device=str(device))

    start_episode = 0
    if args.resume and os.path.exists(ckpt_path):
        trainer.load(ckpt_path)
        start_episode = trainer.total_updates

    # Broadcast initial weights from rank 0
    if worldSize > 1:
        for p in trainer.net.parameters():
            dist.broadcast(p.data, src=0)
        dist.barrier()

    # ── Logger + header (rank 0 only) ─────────────────────────
    logger = None
    if rank == 0:
        logger = MARLLogger(
            algo_name    = f"BR_pred_policy_{args.policy}_{prey_policy.name}",
            log_interval = LOG_INTERVAL,
            seed         = args.policy,
            log_file     = os.path.join(CKPT_DIR, f"br_policy_{args.policy}_metrics.jsonl"),
        )
        print(f"\nTraining BR predator vs prey policy {args.policy}: {prey_policy.name}")
        print(f"Workers: {worldSize}  |  Total episodes: {args.maxEpisodes}")
        print(f"{'Ep':>5} {'Steps':>7} {'Return':>8} {'Win%':>8} {'PLoss':>8} {'VLoss':>8} {'EpSec':>6}")
        print("-" * 58)

    total_steps = 0
    win_history = []
    episodes_this_worker = args.maxEpisodes // worldSize

    for episode in range(start_episode, episodes_this_worker):
        t0        = time.time()
        obs_all   = env.reset()
        done      = False
        ep_return = 0.0

        last_actions = [
            (2, 0.0, 1),
            (2, 0.0, 1),
            (2, 2,   0),
        ]

        last_losses = {"actor_loss": 0.0, "critic_loss": 0.0}

        while not done:
            state_self = encode_obs(SELF_AGENT_IDX, obs_all, last_actions)
            state_team = encode_obs(TEAMMATE_IDX,   obs_all, last_actions)

            self_act, log_prob, value = trainer.select_action(state_self)
            team_act, _,        _    = trainer.select_action(state_team)

            move0, turn_bin0, attack0 = decode_pred_action_30(self_act)
            move1, turn_bin1, attack1 = decode_pred_action_30(team_act)
            prey_move, prey_turn      = prey_policy(obs_all[PREY_IDX], PREY_IDX)

            actions = [
                (move0, turn_bin5_to_cont(turn_bin0), attack0),
                (move1, turn_bin5_to_cont(turn_bin1), attack1),
                (prey_move, prey_turn, 0),
            ]

            obs_next, rewards, dones = env.step(actions)
            self_reward = rewards[SELF_AGENT_IDX]
            done        = bool(dones[SELF_AGENT_IDX])

            trainer.store(state_self, self_act, log_prob, self_reward, value, float(done))
            ep_return   += self_reward
            total_steps += worldSize
            obs_all      = obs_next
            last_actions = actions

            if trainer.buffer.is_full():
                if not done:
                    ns       = encode_obs(SELF_AGENT_IDX, obs_all, last_actions)
                    _, _, lv = trainer.select_action(ns)
                else:
                    lv = 0.0

                last_losses = trainer.update(last_value=lv)

                if worldSize > 1:
                    _allReduceGrads(list(trainer.net.parameters()), worldSize)
                    dist.barrier()

        if len(trainer.buffer) > 0:
            last_losses = trainer.update(last_value=0.0)
            if worldSize > 1:
                _allReduceGrads(list(trainer.net.parameters()), worldSize)
                dist.barrier()

        # ── Logging (rank 0 only) ──────────────────────────────
        if rank == 0:
            ep_sec  = time.time() - t0
            tagged  = env.preyWasTagged
            win_history.append(1 if tagged else 0)
            win_pct = 100.0 * sum(win_history[-100:]) / min(len(win_history), 100)

            print(f"{episode+1:5d} {total_steps:7d} {ep_return:8.1f} {win_pct:8.2f} "
                  f"{last_losses['actor_loss']:8.4f} {last_losses['critic_loss']:8.4f} {ep_sec:6.1f}")

            logger.log_episode(
                episode        = episode + 1,
                episode_return = ep_return,
                win            = int(tagged),
            )

            if (episode + 1) % SAVE_INTERVAL == 0:
                trainer.save(ckpt_path)

    if rank == 0:
        trainer.save(ckpt_path)
        logger.print_final_summary()

    if worldSize > 1:
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train BR predator for OMIS (distributed)")
    parser.add_argument("--policy",      type=int,  default=0,   help="Prey policy index (0-9)")
    parser.add_argument("--numWorkers",  type=int,  default=1,   help="Number of parallel workers")
    parser.add_argument("--maxEpisodes", type=int,  default=500, help="Total episodes across all workers")
    parser.add_argument("--resume",      action="store_true",    help="Resume from checkpoint")
    args = parser.parse_args()

    if args.numWorkers > 1:
        rendFile = os.path.join(
            tempfile.gettempdir(),
            f"br_dist_policy{args.policy}.rend",
        )
        if os.path.exists(rendFile):
            os.remove(rendFile)
        args.rendFile = rendFile
        mp.spawn(workerFn, args=(args.numWorkers, args), nprocs=args.numWorkers, join=True)
    else:
        args.rendFile = ""
        workerFn(0, 1, args)

    print(f"\n[trainBR] Done. Policy {args.policy} → {CKPT_DIR}")
    print("Next step: run experiments/trainOMIS.py")


if __name__ == "__main__":
    main()
