"""
evalOMIS.py
===========
Dual-predator evaluation for shared OMIS model.

- Predator1 and Predator2 both use the same OMIS checkpoint
- Each predator has its own ContextBuilder
- Prey uses scripted policies from Pi_test_prey (or Pi_train_prey for seen)
"""

import os
import sys
import argparse
import json
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from envs.malmoEnvOmis import MalmoEnv
from models.voxelEncoder import VoxelEncoder
from agents.omisAgent import (
    OMISModel,
    ContextBuilder,
    DecisionTimeSearch,
    decode_self_action,
    undiscretize_turn,
    encode_prey_action,
    H_EPI,
    C_EPI,
)
from utils.obsUtils import flattenObs
from utils.scriptedPolicies import get_pi_train_prey, get_pi_test_prey
from utils.logs import MARLLogger


MISSION_XML = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
OMIS_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "omis")
OMIS_CKPT = os.path.join(OMIS_CKPT_DIR, "omis_pretrained.pt")

TOTAL_EVAL_EPISODES = 40
E_SWITCH = 10
M_ROLLOUTS = 3
L_ROLLOUT = 3
GAMMA_SEARCH = 0.7
EPSILON_MIX = 10.0

PREDATOR_INDICES = [0, 1]
PREY_IDX = 2


class MalmoEnvModel:
    """Placeholder environment model for DTS rollouts."""

    def __init__(self, env, voxel_encoder, device, prey_policy):
        self.env = env
        self.voxel_encoder = voxel_encoder
        self.device = device
        self.prey_policy = prey_policy

    def set_opponent_policy(self, policy):
        self.prey_policy = policy

    def __call__(self, state, self_act_flat, pred_opp_act_flat, prey_opp_act_flat):
        _ = self_act_flat
        _ = pred_opp_act_flat
        _ = prey_opp_act_flat
        reward = 0.1
        next_state = state
        return next_state, reward


def encode_obs(agent_idx, obs_all, last_actions, voxel_encoder, device):
    """Encode one predator observation into 128-dim state."""
    flat = flattenObs(agent_idx, obs_all[agent_idx], obs_all, last_actions)
    flat_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        state = voxel_encoder(flat_t)
    return state.squeeze(0).cpu().numpy()


def _pred_action_to_env_tuple(flat_idx: int):
    move, turn_bin, attack = decode_self_action(flat_idx)
    return move, undiscretize_turn(turn_bin), attack


def run_eval_episode(
    env,
    model,
    dts,
    voxel_encoder,
    context_0,
    context_1,
    prey_policy,
    device,
    use_dts=True,
):
    obs_all = env.reset()

    context_0.reset_step_context()
    context_1.reset_step_context()

    total_reward_0 = 0.0
    total_reward_1 = 0.0

    pred_opp_correct = 0
    prey_opp_correct = 0
    opp_total = 0

    search_count = 0
    total_decisions = 0

    done = False
    max_steps = getattr(env, "EPISODE_STEP_LIMIT", 500)

    last_actions = [
        (2, 0.0, 1),
        (2, 0.0, 1),
        (2, 2, 0),
    ]

    for t in range(max_steps):
        if done:
            break

        state_0 = encode_obs(0, obs_all, last_actions, voxel_encoder, device)
        state_1 = encode_obs(1, obs_all, last_actions, voxel_encoder, device)

        if use_dts:
            act_0, _, mode_0 = dts.select_action(context_0, state_0, t)
            act_1, _, mode_1 = dts.select_action(context_1, state_1, t)
            search_count += int(mode_0 == "search") + int(mode_1 == "search")
            total_decisions += 2
        else:
            ctx0 = context_0.get_input_tensors(state_0, t)
            ctx1 = context_1.get_input_tensors(state_1, t)
            with torch.no_grad():
                act_0, _ = model.get_action(**ctx0)
                act_1, _ = model.get_action(**ctx1)
            act_0 = int(act_0.item())
            act_1 = int(act_1.item())

        prey_move, prey_turn = prey_policy(obs_all[PREY_IDX], PREY_IDX)
        prey_act_idx = encode_prey_action(prey_move, prey_turn)

        # Opponent prediction metrics for both predator perspectives.
        ctx0 = context_0.get_input_tensors(state_0, t)
        ctx1 = context_1.get_input_tensors(state_1, t)
        with torch.no_grad():
            pred_opp_0, prey_opp_0, _, _ = model.get_opp_action(**ctx0)
            pred_opp_1, prey_opp_1, _, _ = model.get_opp_action(**ctx1)

        pred_opp_correct += int(int(pred_opp_0.item()) == int(act_1))
        pred_opp_correct += int(int(pred_opp_1.item()) == int(act_0))
        prey_opp_correct += int(int(prey_opp_0.item()) == int(prey_act_idx))
        prey_opp_correct += int(int(prey_opp_1.item()) == int(prey_act_idx))
        opp_total += 2

        actions = [
            _pred_action_to_env_tuple(int(act_0)),
            _pred_action_to_env_tuple(int(act_1)),
            (prey_move, prey_turn, 0),
        ]

        obs_next, rewards, dones = env.step(actions)
        done = bool(dones[0])

        total_reward_0 += rewards[0]
        total_reward_1 += rewards[1]

        context_0.add_step(
            state=state_0,
            self_act=int(act_0),
            pred_opp=int(act_1),
            prey_opp=int(prey_act_idx),
            rtg=float(rewards[0]),
            t=t,
        )
        context_1.add_step(
            state=state_1,
            self_act=int(act_1),
            pred_opp=int(act_0),
            prey_opp=int(prey_act_idx),
            rtg=float(rewards[1]),
            t=t,
        )

        obs_all = obs_next
        last_actions = actions

    avg_return = (total_reward_0 + total_reward_1) / 2.0
    return {
        "episode_return": avg_return,
        "win": int(env.preyWasTagged),
        "pred_opp_acc": pred_opp_correct / max(opp_total, 1),
        "prey_opp_acc": prey_opp_correct / max(opp_total, 1),
        "search_rate": search_count / max(total_decisions, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate shared OMIS (dual predator)")
    parser.add_argument("--no_dts", action="store_true", help="Disable DTS")
    parser.add_argument("--seen_only", action="store_true", help="Use only seen prey policies")
    parser.add_argument("--unseen_only", action="store_true", help="Use only unseen prey policies")
    parser.add_argument("--e_switch", type=int, default=E_SWITCH)
    parser.add_argument("--n_episodes", type=int, default=TOTAL_EVAL_EPISODES)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    use_dts = not args.no_dts
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    algo = "OMIS" if use_dts else "OMIS_w_o_Search"

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {algo}")
    print(f"DTS: {use_dts} | E_switch: {args.e_switch} | Episodes: {args.n_episodes}")
    print(f"{'=' * 60}\n")

    if not os.path.exists(OMIS_CKPT):
        print(f"ERROR: OMIS checkpoint not found: {OMIS_CKPT}")
        print("Run trainOMIS.py first.")
        return

    model = OMISModel().to(device)
    ckpt = torch.load(OMIS_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env = MalmoEnv(MISSION_XML)
    voxel_encoder = VoxelEncoder().to(device)
    voxel_encoder.eval()

    pi_train_prey = get_pi_train_prey()
    pi_test_prey = get_pi_test_prey()

    if args.seen_only:
        eval_configs = [("seen", p) for p in pi_train_prey]
    elif args.unseen_only:
        eval_configs = [("unseen", p) for p in pi_test_prey]
    else:
        eval_configs = [("seen", p) for p in pi_train_prey] + [("unseen", p) for p in pi_test_prey]

    env_model = MalmoEnvModel(env, voxel_encoder, str(device), eval_configs[0][1])
    dts = DecisionTimeSearch(
        model=model,
        env_model=env_model,
        M=M_ROLLOUTS,
        L=L_ROLLOUT,
        gamma_search=GAMMA_SEARCH,
        epsilon=EPSILON_MIX,
        device=str(device),
    )

    logger = MARLLogger(algo_name=algo, log_interval=1)

    n_switches = max(1, args.n_episodes // args.e_switch)
    global_episode = 0

    # Store trajectories per policy bucket for D_epi construction.
    recent_trajs = {idx: [] for idx in range(len(eval_configs))}

    for switch_idx in range(n_switches):
        config_idx = switch_idx % len(eval_configs)
        opp_type, prey_policy = eval_configs[config_idx]

        env_model.set_opponent_policy(prey_policy)

        context_0 = ContextBuilder(device=str(device))
        context_1 = ContextBuilder(device=str(device))

        if len(recent_trajs[config_idx]) >= C_EPI:
            context_0.set_epi_context(recent_trajs[config_idx][-C_EPI:])
            context_1.set_epi_context(recent_trajs[config_idx][-C_EPI:])

        for _ in range(args.e_switch):
            if global_episode >= args.n_episodes:
                break

            metrics = run_eval_episode(
                env,
                model,
                dts,
                voxel_encoder,
                context_0,
                context_1,
                prey_policy,
                str(device),
                use_dts=use_dts,
            )

            if context_0.step_states:
                traj0 = list(
                    zip(
                        context_0.step_states,
                        context_0.step_pred_opp_acts,
                        context_0.step_prey_opp_acts,
                    )
                )
                recent_trajs[config_idx].append(traj0)

            if context_1.step_states:
                traj1 = list(
                    zip(
                        context_1.step_states,
                        context_1.step_pred_opp_acts,
                        context_1.step_prey_opp_acts,
                    )
                )
                recent_trajs[config_idx].append(traj1)

            if len(recent_trajs[config_idx]) > 20:
                recent_trajs[config_idx] = recent_trajs[config_idx][-20:]

            logger.log_episode(
                episode=global_episode + 1,
                episode_return=metrics["episode_return"],
                win=metrics["win"],
                opponent_type=opp_type,
            )
            logger.log_eval(
                episode=global_episode + 1,
                episode_return=metrics["episode_return"],
                win=metrics["win"],
                pred_opp_acc=metrics["pred_opp_acc"],
                prey_opp_acc=metrics["prey_opp_acc"],
                search_rate=metrics["search_rate"],
                opponent_type=opp_type,
            )

            global_episode += 1

        print(
            f"  Policy: {prey_policy.name:<25} | Type: {opp_type:<8} | Episode {global_episode}",
            flush=True,
        )

    logger.print_eval_summary(episode=global_episode)
    logger.print_final_summary()

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "timestamp": timestamp,
        "algorithm": algo,
        "use_dts": use_dts,
        "e_switch": args.e_switch,
        "total_episodes": global_episode,
        "dts_config": {
            "m_rollouts": M_ROLLOUTS,
            "l_rollout": L_ROLLOUT,
            "gamma_search": GAMMA_SEARCH,
            "epsilon_mix": EPSILON_MIX,
        },
    }

    json_path = os.path.join(results_dir, f"eval_results_{algo}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    txt_path = os.path.join(results_dir, f"eval_results_{algo}_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write(f"EVALUATION RESULTS - {algo}\n")
        file.write("=" * 80 + "\n\n")
        file.write(f"Timestamp: {timestamp}\n")
        file.write(f"Algorithm: {algo}\n")
        file.write(f"Use DTS: {use_dts}\n")
        file.write(f"Total episodes: {global_episode}\n")
        file.write(f"Policy switching (E): {args.e_switch}\n")

    print("\n[evalOMIS] Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  TXT:  {txt_path}")
    print("[evalOMIS] Evaluation complete!")


if __name__ == "__main__":
    main()
