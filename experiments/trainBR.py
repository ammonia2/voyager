"""
trainBR.py
==========
Phase 1 of OMIS pipeline: Train Best Response (BR) agents.

For each scripted policy π⁻¹,k in Pi_train (k = 0..9):
  - Fix both predators to use scripted policy k
  - Train a PPO agent as the prey (self-agent, index 2)
  - Save the trained BR_k checkpoint

This must be run BEFORE dataCollector and trainOMIS.

Usage:
    conda activate marl-malmo
    # Start 3 Minecraft clients on ports 10000, 10001, 10002
    python experiments/trainBR.py
    python experiments/trainBR.py --policy 3   # train only policy 3
    python experiments/trainBR.py --resume      # resume from checkpoint

Hyperparameters (Appendix H.2):
    Total episodes : 50,000 per policy
    Batch size     : 4096 steps
    PPO clip       : 0.2
    LR actor/critic: 5e-4
    Discount γ     : 1.0
    Grad clip      : 5.0
    Hidden dim     : 32, 3 layers
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from envs.malmoEnvOmis          import MalmoEnv
from models.voxelEncoder    import VoxelEncoder
from agents.brAgent         import PPOTrainer, TOTAL_EPISODES, BATCH_SIZE
from utils.scriptedPolicies import get_policy_by_index, get_flat_action
from utils.logs              import MARLLogger

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

MISSION_XML     = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
CKPT_DIR        = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "br")
LOG_INTERVAL    = 1    # print every N episodes
SAVE_INTERVAL   = 1000   # save checkpoint every N episodes
SELF_AGENT_IDX  = 2      # prey
OPP_INDICES     = [0, 1] # predators


# ─────────────────────────────────────────────────────────────────
# Observation → state vector
# ─────────────────────────────────────────────────────────────────

def encode_obs(obs, voxel_encoder, device):
    """Convert raw Malmo obs dict → 128-dim state tensor (numpy)."""
    voxel_grid = torch.tensor(
        obs.get("ob", [0] * 49), dtype=torch.long
    ).unsqueeze(0).to(device)

    entities_raw = obs.get("entities", [])
    max_ents     = 4
    ent_tensor   = torch.zeros(1, max_ents, 5, device=device)
    ent_mask     = torch.zeros(1, max_ents, device=device)
    for i, e in enumerate(entities_raw[:max_ents]):
        ent_tensor[0, i] = torch.tensor(e[:5], dtype=torch.float32)
        ent_mask[0, i]   = 1.0

    stats = torch.tensor(
        [obs.get("x", 0.0), obs.get("z", 0.0),
         obs.get("yaw", 0.0), obs.get("life", 20.0)],
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        state = voxel_encoder(voxel_grid, ent_tensor, ent_mask, stats)

    return state.squeeze(0).cpu().numpy()   # (128,)


# ─────────────────────────────────────────────────────────────────
# Train one BR policy
# ─────────────────────────────────────────────────────────────────

def train_br(policy_idx, env, voxel_encoder, device, resume=False):
    """
    Train Best Response against scripted policy k.

    Returns the trained PPOTrainer.
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"br_policy_{policy_idx}.pt")

    scripted_policy = get_policy_by_index(policy_idx)
    trainer = PPOTrainer(
        policy_idx = policy_idx,
        device     = device,
    )

    start_episode = 0
    if resume and os.path.exists(ckpt_path):
        trainer.load(ckpt_path)
        start_episode = trainer.total_updates  # rough estimate

    logger = MARLLogger(
        algo_name    = f"BR_policy_{policy_idx}_{scripted_policy.name}",
        log_interval = LOG_INTERVAL,
        seed         = policy_idx,
    )

    print(f"\n{'='*60}", flush=True)
    print(f"Training BR against policy {policy_idx}: {scripted_policy.name}", flush=True)
    print(f"{'='*60}", flush=True)

    global_step = 0

    for episode in range(start_episode, TOTAL_EPISODES):
        obs_all  = env.reset()
        done     = False
        ep_return = 0.0
        step_count = 0

        while not done:
            prey_obs   = obs_all[SELF_AGENT_IDX]
            prey_state = encode_obs(prey_obs, voxel_encoder, device)

            # BR selects action
            self_act, log_prob, value = trainer.select_action(prey_state)

            # Scripted policy for predators
            opp_act0 = get_flat_action(scripted_policy(obs_all[OPP_INDICES[0]], 0))
            opp_act1 = get_flat_action(scripted_policy(obs_all[OPP_INDICES[1]], 1))

            actions  = [opp_act0, opp_act1, self_act]
            obs_next, rewards, dones, _ = env.step(actions)

            prey_reward = rewards[SELF_AGENT_IDX]
            done        = dones[SELF_AGENT_IDX]

            trainer.store(prey_state, self_act, log_prob, prey_reward, value, float(done))

            ep_return  += prey_reward
            global_step += 1
            obs_all     = obs_next

            # Update when buffer is full
            if trainer.buffer.is_full():
                # Bootstrap last value
                if not done:
                    next_state = encode_obs(obs_all[SELF_AGENT_IDX], voxel_encoder, device)
                    _, _, last_val = trainer.select_action(next_state)
                else:
                    last_val = 0.0

                losses = trainer.update(last_value=last_val)
                # logger.log_step(
                #     step         = global_step,
                #     actor_loss   = losses["actor_loss"],
                #     critic_loss  = losses["critic_loss"],
                # )

        # End of episode — flush any remaining buffer
        if len(trainer.buffer) > 0:
            losses = trainer.update(last_value=0.0)

        # Log episode
        logger.log_episode(
            episode        = episode + 1,
            episode_return = ep_return,
            win            = int(not env.preyWasTagged),
        )
        
        # Print progress every episode with immediate output
        if (episode + 1) % 1 == 0:
            print(f"  Policy {policy_idx} | Ep {episode+1} | Return {ep_return:.2f}", flush=True)

        # Save checkpoint
        if (episode + 1) % SAVE_INTERVAL == 0:
            trainer.save(ckpt_path)

    # Final save
    trainer.save(ckpt_path)
    logger.print_final_summary()

    return trainer


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train BR agents for OMIS")
    parser.add_argument("--policy", type=int, default=-1,
                        help="Train only this policy index (default: train all 0-9)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment
    env = MalmoEnv(MISSION_XML)

    # Initialize shared VoxelEncoder (frozen — not trained here)
    voxel_encoder = VoxelEncoder().to(device)
    voxel_encoder.eval()
    for p in voxel_encoder.parameters():
        p.requires_grad = False

    # Determine which policies to train
    if args.policy >= 0:
        policy_indices = [args.policy]
    else:
        policy_indices = list(range(10))

    for k in policy_indices:
        train_br(k, env, voxel_encoder, device, resume=args.resume)
        print(f"\n[trainBR] Completed BR for policy {k}")

    print("\n[trainBR] All BR agents trained successfully.")
    print(f"Checkpoints saved to: {CKPT_DIR}")
    print("Next step: run experiments/trainOMIS.py")


if __name__ == "__main__":
    main()
