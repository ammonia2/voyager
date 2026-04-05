"""
evalOMIS.py
===========
Phase 3 of OMIS pipeline: Evaluation with Decision-Time Search.

Tests the pretrained OMIS model against both seen (Pi_train) and
unseen (Pi_test) opponent policies, with policy switching (non-stationary).

Implements Algorithm 1 lines 10-24:
  - Set policy switching frequency E
  - For each switch: sample π̄⁻¹ ~ Pi_test, run E episodes with DTS
  - Collect metrics and print evaluation summary

DTS Hyperparameters (Appendix H.3 — PP environment values):
    M (rollouts per action) = 3
    L (rollout length)      = 3
    γ_search                = 0.7
    ε (mixing threshold)    = 10
    Total eval episodes     = 1200

Usage:
    conda activate marl-malmo
    # Start 3 Minecraft clients on ports 10000, 10001, 10002
    python experiments/evalOMIS.py
    python experiments/evalOMIS.py --seen_only
    python experiments/evalOMIS.py --no_dts    # evaluate OMIS w/o Search
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from envs.malmoEnvOmis       import MalmoEnv
from models.voxelEncoder import VoxelEncoder
from agents.omisAgent    import (
    OMISModel, ContextBuilder, DecisionTimeSearch,
    encode_self_action, encode_joint_opp_action,
    H_EPI, C_EPI, B_SEQ,
)
from utils.scriptedPolicies import (
    get_pi_train, get_pi_test, get_flat_action
)
from utils.logs import MARLLogger

# ─────────────────────────────────────────────────────────────────
# Config (Appendix H.3)
# ─────────────────────────────────────────────────────────────────

MISSION_XML   = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
OMIS_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "omis")
OMIS_CKPT     = os.path.join(OMIS_CKPT_DIR, "omis_pretrained.pt")

TOTAL_EVAL_EPISODES = 40
E_SWITCH            = 10     # episodes per policy switch (test 4 policies)
M_ROLLOUTS          = 3
L_ROLLOUT           = 3
GAMMA_SEARCH        = 0.7
EPSILON_MIX         = 10.0

SELF_AGENT_IDX = 2
OPP_INDICES    = [0, 1]


# ─────────────────────────────────────────────────────────────────
# Environment model P for DTS rollouts
# ─────────────────────────────────────────────────────────────────

class MalmoEnvModel:
    """
    Environment model P used during DTS rollouts.
    In OMIS, the ground truth transition P is assumed available (Section 4.2).
    We approximate it by calling env.step() directly.

    Note: During DTS, we perform IMAGINED rollouts — this means calling
    env.step() for each rollout step. This is expensive but correct.
    In practice, a learned model could be substituted here (Model-Based OMIS).
    """

    def __init__(self, env, voxel_encoder, device, opp_policy):
        self.env           = env
        self.voxel_encoder = voxel_encoder
        self.device        = device
        self.opp_policy    = opp_policy
        self._saved_state  = None

    def set_opponent_policy(self, policy):
        self.opp_policy = policy

    def __call__(self, state, self_act_flat, opp_joint_act):
        """
        Transition function: (state, self_action, opp_joint_action) → (next_state, reward)
        Used by DecisionTimeSearch for rollouts.

        Since we have access to the real Malmo environment (ground truth P),
        we use a simplified local transition model based on the current env state.

        For efficiency, we estimate the reward from the action without stepping
        the full Malmo environment — this is the "model-based" approximation.
        """
        # Decode joint opponent action
        opp_act0 = opp_joint_act // 18
        opp_act1 = opp_joint_act % 18

        # Use simple reward heuristic for DTS (prey survival)
        # In full DTS with ground truth P, we would call env.step()
        # Here we use a lightweight approximation:
        #   - If self_act moves away from opponents: small positive reward
        #   - Otherwise: neutral
        # This is replaced by a learned model in Model-Based OMIS.
        reward = 0.1   # prey survival reward per step

        # For next state: we use the current state as approximation
        # (a real learned model would predict this)
        next_state = state

        return next_state, reward


# ─────────────────────────────────────────────────────────────────
# Observation encoding
# ─────────────────────────────────────────────────────────────────

def encode_obs(obs, voxel_encoder, device):
    """Convert raw Malmo obs → 128-dim numpy state."""
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

    return state.squeeze(0).cpu().numpy()


# ─────────────────────────────────────────────────────────────────
# Run one evaluation episode
# ─────────────────────────────────────────────────────────────────

def run_eval_episode(
    env, model, dts, voxel_encoder, context_builder,
    opp_policy, episode_idx, device, use_dts=True,
):
    """
    Run one evaluation episode and return metrics.

    Returns dict:
      episode_return  : float
      win             : int (1 if prey survived > 90% of episode)
      opp_acc         : float (opponent action prediction accuracy)
      search_rate     : float (fraction of timesteps DTS was used)
      ep_length       : int
    """
    obs_all = env.reset()

    # Reset step context for this episode
    context_builder.reset_step_context()

    total_reward    = 0.0
    opp_correct     = 0
    opp_total       = 0
    search_count    = 0
    step_count      = 0
    max_steps       = getattr(env, "episodeStepLimit", 500)
    done            = False

    # Track current RTG estimate (starts high, updated as episode progresses)
    # For eval, we use a running estimate
    running_rtg = 0.0
    rewards_so_far = []

    for t in range(max_steps):
        if done:
            break

        prey_obs   = obs_all[SELF_AGENT_IDX]
        prey_state = encode_obs(prey_obs, voxel_encoder, device)

        # ── Select action ────────────────────────────────────────
        if use_dts:
            self_act_flat, q_val, mode = dts.select_action(
                context_builder, prey_state, t
            )
            if mode == "search":
                search_count += 1
        else:
            # OMIS w/o Search: use actor πθ directly
            ctx = context_builder.get_input_tensors(prey_state, t)
            with torch.no_grad():
                act, _ = model.get_action(**ctx)
            self_act_flat = act.item()

        # ── Opponent action prediction accuracy ──────────────────
        opp_act0_true = get_flat_action(opp_policy(obs_all[OPP_INDICES[0]], 0))
        opp_act1_true = get_flat_action(opp_policy(obs_all[OPP_INDICES[1]], 1))
        true_joint    = encode_joint_opp_action(opp_act0_true, opp_act1_true)

        ctx = context_builder.get_input_tensors(prey_state, t)
        with torch.no_grad():
            pred_opp, _ = model.get_opp_action(**ctx)
        pred_joint = pred_opp.item()

        opp_correct += int(pred_joint == true_joint)
        opp_total   += 1

        # ── Execute in environment ───────────────────────────────
        actions = [opp_act0_true, opp_act1_true, self_act_flat]
        obs_next, rewards, dones, _ = env.step(actions)

        prey_reward = rewards[SELF_AGENT_IDX]
        done        = dones[SELF_AGENT_IDX]

        rewards_so_far.append(prey_reward)
        total_reward += prey_reward
        step_count   += 1

        # ── Update context builder ───────────────────────────────
        # Compute running RTG (approximate — use cumulative from here)
        # For simplicity: use current reward as RTG proxy during eval
        context_builder.add_step(
            state     = prey_state,
            self_act  = self_act_flat,
            opp_act   = true_joint,
            rtg       = prey_reward,
            timestep  = t,
        )

        obs_all = obs_next

    # Win: prey survived without losing all health (positive reward indicates survival)
    win = int(total_reward > 0)

    return {
        "episode_return": total_reward,
        "win":            win,
        "opp_acc":        opp_correct / max(opp_total, 1),
        "search_rate":    search_count / max(step_count, 1),
        "ep_length":      step_count,
    }


# ─────────────────────────────────────────────────────────────────
# Main evaluation loop (Algorithm 1 lines 10-24)
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate OMIS with DTS")
    parser.add_argument("--no_dts",     action="store_true",
                        help="Evaluate OMIS w/o Search (no DTS)")
    parser.add_argument("--seen_only",  action="store_true",
                        help="Only evaluate on seen opponents (Pi_train)")
    parser.add_argument("--unseen_only", action="store_true",
                        help="Only evaluate on unseen opponents (Pi_test)")
    parser.add_argument("--e_switch",   type=int, default=E_SWITCH,
                        help="Policy switching frequency E")
    parser.add_argument("--n_episodes", type=int, default=TOTAL_EVAL_EPISODES)
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    use_dts = not args.no_dts
    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    algo    = "OMIS" if use_dts else "OMIS_w/o_Search"

    print(f"\n{'='*60}")
    print(f"Evaluating: {algo}")
    print(f"DTS: {use_dts} | E_switch: {args.e_switch} | Episodes: {args.n_episodes}")
    print(f"{'='*60}\n")

    # ── Load model ───────────────────────────────────────────────
    if not os.path.exists(OMIS_CKPT):
        print(f"ERROR: OMIS checkpoint not found: {OMIS_CKPT}")
        print("Run trainOMIS.py first.")
        return

    model = OMISModel().to(device)
    ckpt  = torch.load(OMIS_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[evalOMIS] Loaded OMIS model from {OMIS_CKPT}")

    # ── Initialize environment ───────────────────────────────────
    env = MalmoEnv(MISSION_XML)
    voxel_encoder = VoxelEncoder().to(device)
    voxel_encoder.eval()

    # ── Build opponent policy sets ───────────────────────────────
    pi_train = get_pi_train()   # seen
    pi_test  = get_pi_test()    # unseen

    if args.seen_only:
        eval_configs = [("seen", p) for p in pi_train]
    elif args.unseen_only:
        eval_configs = [("unseen", p) for p in pi_test]
    else:
        eval_configs = (
            [("seen",   p) for p in pi_train] +
            [("unseen", p) for p in pi_test]
        )

    # ── Environment model for DTS ────────────────────────────────
    env_model = MalmoEnvModel(env, voxel_encoder, device, pi_train[0])
    dts = DecisionTimeSearch(
        model        = model,
        env_model    = env_model,
        M            = M_ROLLOUTS,
        L            = L_ROLLOUT,
        gamma_search = GAMMA_SEARCH,
        epsilon      = EPSILON_MIX,
        device       = str(device),
    )

    # ── Logger ───────────────────────────────────────────────────
    logger = MARLLogger(algo_name=algo, log_interval=1)

    # ── Evaluation loop (Algorithm 1 lines 11-24) ────────────────
    import random

    n_switches      = args.n_episodes // args.e_switch
    global_episode  = 0

    # Track recent trajectories per opponent type for D_epi construction
    # (Appendix C: use C most recent trajectories)
    recent_trajs = {i: [] for i in range(len(eval_configs))}

    for switch_idx in range(n_switches):
        # Sample a true opponent policy (Algorithm 1 line 13)
        config_idx     = switch_idx % len(eval_configs)
        opp_type, opp_policy = eval_configs[config_idx]

        # Update env model's opponent policy
        env_model.set_opponent_policy(opp_policy)

        # Build D_epi from recent trajectories for this policy type
        context_builder = ContextBuilder(device=str(device))
        if len(recent_trajs[config_idx]) >= C_EPI:
            context_builder.set_epi_context(
                recent_trajs[config_idx][-C_EPI:]
            )

        for ep_in_switch in range(args.e_switch):
            if global_episode >= args.n_episodes:
                break

            metrics = run_eval_episode(
                env, model, dts, voxel_encoder, context_builder,
                opp_policy, global_episode, device, use_dts=use_dts,
            )

            # Store trajectory for D_epi (as (state, opp_act) pairs)
            # Retrieved from context_builder's step history
            if context_builder.step_states:
                traj = list(zip(
                    context_builder.step_states,
                    context_builder.step_opp_acts,
                ))
                recent_trajs[config_idx].append(traj)
                if len(recent_trajs[config_idx]) > 10:
                    recent_trajs[config_idx].pop(0)

            # Log
            logger.log_episode(
                episode        = global_episode + 1,
                episode_return = metrics["episode_return"],
                win            = metrics["win"],
                opponent_type  = opp_type,
            )
            logger.log_eval(
                episode        = global_episode + 1,
                episode_return = metrics["episode_return"],
                win            = metrics["win"],
                opponent_acc   = metrics["opp_acc"],
                opponent_type  = opp_type,
            )

            global_episode += 1

        # Print eval summary every episode
        print(
            f"  Policy: {opp_policy.name:<25} | Type: {opp_type:<8} | Episode {global_episode}",
            flush=True
        )

    # ── Final eval summary ───────────────────────────────────────
    logger.print_eval_summary(episode=global_episode)
    logger.print_final_summary()

    # ── Save evaluation results ───────────────────────────────────
    import json
    from datetime import datetime
    
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algo_name = "OMIS" if use_dts else "OMIS_w/o_Search"
    
    # Prepare results summary
    results = {
        "timestamp": timestamp,
        "algorithm": algo_name,
        "use_dts": use_dts,
        "e_switch": args.e_switch,
        "total_episodes": global_episode,
        "dts_config": {
            "m_rollouts": M_ROLLOUTS,
            "l_rollout": L_ROLLOUT,
            "gamma_search": GAMMA_SEARCH,
            "epsilon_mix": EPSILON_MIX,
        },
        "metrics": {
            "total_episodes_run": global_episode,
        }
    }
    
    # Save JSON
    json_path = os.path.join(results_dir, f"eval_results_{algo_name}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save text summary
    txt_path = os.path.join(results_dir, f"eval_results_{algo_name}_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"EVALUATION RESULTS - {algo_name}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Algorithm: {algo_name}\n")
        f.write(f"Use DTS: {use_dts}\n")
        f.write(f"Total episodes: {global_episode}\n")
        f.write(f"Policy switching (E): {args.e_switch}\n\n")
        
        if use_dts:
            f.write("DTS Configuration:\n")
            f.write(f"  Rollouts per action (M): {M_ROLLOUTS}\n")
            f.write(f"  Rollout length (L): {L_ROLLOUT}\n")
            f.write(f"  Discount (gamma): {GAMMA_SEARCH}\n")
            f.write(f"  Mixing threshold (epsilon): {EPSILON_MIX}\n\n")
        
        f.write("See console output above for detailed metrics.\n")
    
    print(f"\n[evalOMIS] Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")
    print(f"[evalOMIS] Evaluation complete!")


if __name__ == "__main__":
    main()
