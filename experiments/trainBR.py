"""
trainBR.py
==========
Phase 1 of OMIS pipeline (predator-overhaul version).

Train BR predator policies against fixed scripted prey evasion policies.
Learner: Predator1 (index 0)
Teammate: Predator2 (index 1) using same BR network (self-play style)
Opponent: Prey (index 2) scripted by policy k
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from envs.malmoEnvOmis import MalmoEnv
from models.voxelEncoder import VoxelEncoder
from agents.brAgent import PPOTrainer, TOTAL_EPISODES
from utils.obsUtils import flattenObs
from utils.scriptedPolicies import (
    get_prey_policy_by_index,
    decode_pred_action_30,
    turn_bin5_to_cont,
)
from utils.logs import MARLLogger


MISSION_XML = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "br")

LOG_INTERVAL = 1
SAVE_INTERVAL = 1000

SELF_AGENT_IDX = 0
TEAMMATE_IDX = 1
PREY_IDX = 2


def encode_obs(agent_idx, obs_all, last_actions, voxel_encoder, device):
    """Encode one agent's observation into 128-dim state vector."""
    flat = flattenObs(agent_idx, obs_all[agent_idx], obs_all, last_actions)
    flat_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        state = voxel_encoder(flat_t)
    return state.squeeze(0).cpu().numpy()


def train_br(policy_idx, env, voxel_encoder, device, resume=False):
    """Train BR predator against one scripted prey policy."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"br_policy_{policy_idx}.pt")

    prey_policy = get_prey_policy_by_index(policy_idx)
    trainer = PPOTrainer(policy_idx=policy_idx, device=device)

    start_episode = 0
    if resume and os.path.exists(ckpt_path):
        trainer.load(ckpt_path)
        start_episode = trainer.total_updates

    logger = MARLLogger(
        algo_name=f"BR_pred_policy_{policy_idx}_{prey_policy.name}",
        log_interval=LOG_INTERVAL,
        seed=policy_idx,
    )

    print(f"\n{'=' * 60}", flush=True)
    print(f"Training Predator BR against prey policy {policy_idx}: {prey_policy.name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    for episode in range(start_episode, TOTAL_EPISODES):
        obs_all = env.reset()
        done = False
        ep_return = 0.0

        last_actions = [
            (2, 0.0, 1),
            (2, 0.0, 1),
            (2, 2, 0),
        ]

        while not done:
            state_self = encode_obs(SELF_AGENT_IDX, obs_all, last_actions, voxel_encoder, device)
            state_team = encode_obs(TEAMMATE_IDX, obs_all, last_actions, voxel_encoder, device)

            self_act, log_prob, value = trainer.select_action(state_self)
            team_act, _, _ = trainer.select_action(state_team)

            move0, turn_bin0, attack0 = decode_pred_action_30(self_act)
            move1, turn_bin1, attack1 = decode_pred_action_30(team_act)

            prey_move, prey_turn = prey_policy(obs_all[PREY_IDX], PREY_IDX)
            actions = [
                (move0, turn_bin5_to_cont(turn_bin0), attack0),
                (move1, turn_bin5_to_cont(turn_bin1), attack1),
                (prey_move, prey_turn, 0),
            ]

            obs_next, rewards, dones = env.step(actions)

            self_reward = rewards[SELF_AGENT_IDX]
            done = bool(dones[SELF_AGENT_IDX])

            trainer.store(state_self, self_act, log_prob, self_reward, value, float(done))

            ep_return += self_reward
            obs_all = obs_next
            last_actions = actions

            if trainer.buffer.is_full():
                if not done:
                    next_state = encode_obs(SELF_AGENT_IDX, obs_all, last_actions, voxel_encoder, device)
                    _, _, last_val = trainer.select_action(next_state)
                else:
                    last_val = 0.0
                trainer.update(last_value=last_val)

        if len(trainer.buffer) > 0:
            trainer.update(last_value=0.0)

        logger.log_episode(
            episode=episode + 1,
            episode_return=ep_return,
            win=int(env.preyWasTagged),
        )

        print(f"  Policy {policy_idx} | Ep {episode + 1} | Return {ep_return:.2f}", flush=True)

        if (episode + 1) % SAVE_INTERVAL == 0:
            trainer.save(ckpt_path)

    trainer.save(ckpt_path)
    logger.print_final_summary()
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train predator BR agents for OMIS")
    parser.add_argument("--policy", type=int, default=-1, help="Train only this prey policy index")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = MalmoEnv(MISSION_XML)

    voxel_encoder = VoxelEncoder().to(device)
    voxel_encoder.eval()
    for param in voxel_encoder.parameters():
        param.requires_grad = False

    if args.policy >= 0:
        policy_indices = [args.policy]
    else:
        policy_indices = list(range(10))

    for k in policy_indices:
        train_br(k, env, voxel_encoder, device, resume=args.resume)
        print(f"\n[trainBR] Completed BR for prey policy {k}")

    print("\n[trainBR] All BR agents trained successfully.")
    print(f"Checkpoints saved to: {CKPT_DIR}")
    print("Next step: run experiments/trainOMIS.py")


if __name__ == "__main__":
    main()
