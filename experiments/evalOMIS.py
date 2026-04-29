"""
evalOMIS.py
===========
Dual-predator evaluation for shared OMIS model — parallel workers.

Each worker runs its own Malmo arena independently (no gradient sync
needed since this is pure inference). Results are merged at the end.

Run:
  python experiments/evalOMIS.py --numWorkers 1              # single
  python experiments/evalOMIS.py --numWorkers 2 --n_episodes 500
"""

import os
import sys
import argparse
import json
import time
import multiprocessing as mp
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.malmoEnvOmis import MalmoEnv
from src.agents.omisAgent import (
    OMISModel,
    ContextBuilder,
    DecisionTimeSearch,
    decode_self_action,
    undiscretize_turn,
    encode_prey_action,
    H_EPI,
    C_EPI,
)
from src.utils.obsUtils import flattenObs
from src.utils.scriptedPolicies import get_pi_train_prey, get_pi_test_prey
from src.utils.logs import MARLLogger


MISSION_XML   = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
OMIS_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "omis")
OMIS_CKPT     = os.path.join(OMIS_CKPT_DIR, "omis_pretrained.pt")

TOTAL_EVAL_EPISODES = 500
E_SWITCH     = 10
M_ROLLOUTS   = 3
L_ROLLOUT    = 3
GAMMA_SEARCH = 0.7
EPSILON_MIX  = 10.0

PREDATOR_INDICES = [0, 1]
PREY_IDX         = 2
PORTS_PER_WORKER = 3


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

class MalmoEnvModel:
    """Placeholder environment model for DTS rollouts."""
    def __init__(self, env, device, prey_policy):
        self.env         = env
        self.device      = device
        self.prey_policy = prey_policy

    def set_opponent_policy(self, policy):
        self.prey_policy = policy

    def __call__(self, state, self_act_flat, pred_opp_act_flat, prey_opp_act_flat):
        return state, 0.1


def encode_obs(agent_idx, obs_all, last_actions):
    flat = flattenObs(agent_idx, obs_all[agent_idx], obs_all, last_actions)
    return torch.tensor(flat, dtype=torch.float32)


def _pred_action_to_env_tuple(flat_idx: int):
    move, turn_bin, attack = decode_self_action(flat_idx)
    return move, undiscretize_turn(turn_bin), attack


# ─────────────────────────────────────────────────────────────────
# Single-episode evaluation
# ─────────────────────────────────────────────────────────────────

def run_eval_episode(env, model, dts, context_0, context_1, prey_policy, device, use_dts=True):
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

    last_actions = [(2, 0.0, 1), (2, 0.0, 1), (2, 2, 0)]

    for t in range(max_steps):
        if done:
            break

        state_0 = encode_obs(0, obs_all, last_actions)
        state_1 = encode_obs(1, obs_all, last_actions)

        if use_dts:
            act_0, _, mode_0 = dts.select_action(context_0, state_0.numpy(), t)
            act_1, _, mode_1 = dts.select_action(context_1, state_1.numpy(), t)
            search_count    += int(mode_0 == "search") + int(mode_1 == "search")
            total_decisions += 2
        else:
            ctx0 = context_0.get_input_tensors(state_0.numpy(), t)
            ctx1 = context_1.get_input_tensors(state_1.numpy(), t)
            with torch.no_grad():
                act_0, _ = model.get_action(**ctx0)
                act_1, _ = model.get_action(**ctx1)
            act_0 = int(act_0.item())
            act_1 = int(act_1.item())

        prey_move, prey_turn = prey_policy(obs_all[PREY_IDX], PREY_IDX)
        prey_act_idx = encode_prey_action(prey_move, prey_turn)

        ctx0 = context_0.get_input_tensors(state_0.numpy(), t)
        ctx1 = context_1.get_input_tensors(state_1.numpy(), t)
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

        context_0.add_step(state=state_0.numpy(), self_act=int(act_0),
                           pred_opp=int(act_1), prey_opp=int(prey_act_idx),
                           rtg=float(rewards[0]), t=t)
        context_1.add_step(state=state_1.numpy(), self_act=int(act_1),
                           pred_opp=int(act_0), prey_opp=int(prey_act_idx),
                           rtg=float(rewards[1]), t=t)

        obs_all      = obs_next
        last_actions = actions

    avg_return = (total_reward_0 + total_reward_1) / 2.0
    return {
        "episode_return": avg_return,
        "win":            int(env.preyWasTagged),
        "pred_opp_acc":   pred_opp_correct / max(opp_total, 1),
        "prey_opp_acc":   prey_opp_correct / max(opp_total, 1),
        "search_rate":    search_count / max(total_decisions, 1),
    }


# ─────────────────────────────────────────────────────────────────
# Worker function (runs in its own process)
# ─────────────────────────────────────────────────────────────────

def workerFn(rank, worldSize, args, result_queue):
    print(f"[Worker {rank}] Starting up...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Worker {rank}] Using device: {device}", flush=True)

    # Load model (each worker loads its own copy)
    model = OMISModel().to(device)
    ckpt  = torch.load(OMIS_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Each worker gets its own Malmo arena
    env       = MalmoEnv(MISSION_XML, portOffset=rank * PORTS_PER_WORKER)
    env_model = MalmoEnvModel(env, str(device), None)

    dts = DecisionTimeSearch(
        model        = model,
        env_model    = env_model,
        M            = M_ROLLOUTS,
        L            = L_ROLLOUT,
        gamma_search = GAMMA_SEARCH,
        epsilon      = EPSILON_MIX,
        device       = str(device),
    )

    pi_train_prey = get_pi_train_prey()
    pi_test_prey  = get_pi_test_prey()

    if args.seen_only:
        eval_configs = [("seen",   p) for p in pi_train_prey]
    elif args.unseen_only:
        eval_configs = [("unseen", p) for p in pi_test_prey]
    else:
        eval_configs = (
            [("seen",   p) for p in pi_train_prey] +
            [("unseen", p) for p in pi_test_prey]
        )

    use_dts = not args.no_dts
    n_episodes_this_worker = args.n_episodes // worldSize
    n_switches = max(1, n_episodes_this_worker // args.e_switch)

    recent_trajs  = {idx: [] for idx in range(len(eval_configs))}
    all_metrics   = []
    global_episode = 0

    for switch_idx in range(n_switches):
        config_idx           = switch_idx % len(eval_configs)
        opp_type, prey_policy = eval_configs[config_idx]
        env_model.set_opponent_policy(prey_policy)

        context_0 = ContextBuilder(device=str(device))
        context_1 = ContextBuilder(device=str(device))

        if len(recent_trajs[config_idx]) >= C_EPI:
            context_0.set_epi_context(recent_trajs[config_idx][-C_EPI:])
            context_1.set_epi_context(recent_trajs[config_idx][-C_EPI:])

        for _ in range(args.e_switch):
            if global_episode >= n_episodes_this_worker:
                break

            print(f"[Worker {rank}] Starting episode {global_episode+1}/{n_episodes_this_worker} against {opp_type} prey...", flush=True)
            t0      = time.time()
            metrics = run_eval_episode(env, model, dts, context_0, context_1,
                                       prey_policy, str(device), use_dts=use_dts)

            # Store trajectory for episodic context
            for ctx in (context_0, context_1):
                if ctx.step_states:
                    traj = list(zip(ctx.step_states, ctx.step_pred_opp_acts, ctx.step_prey_opp_acts))
                    recent_trajs[config_idx].append(traj)
            if len(recent_trajs[config_idx]) > 20:
                recent_trajs[config_idx] = recent_trajs[config_idx][-20:]

            metrics["ep_sec"]    = time.time() - t0
            metrics["opp_type"]  = opp_type
            metrics["rank"]      = rank
            all_metrics.append(metrics)
            
            # Print progress from rank 0 so the user sees something
            if rank == 0:
                win_history = [m["win"] for m in all_metrics]
                win_pct = 100.0 * sum(win_history[-100:]) / min(len(win_history), 100)
                print(f"{len(all_metrics):5d} {metrics['episode_return']:8.1f} {win_pct:8.2f} "
                      f"{metrics['pred_opp_acc']:8.4f} {metrics['prey_opp_acc']:8.4f} "
                      f"{metrics['search_rate']:8.4f} {metrics['ep_sec']:6.1f}", flush=True)

            global_episode += 1

    result_queue.put(all_metrics)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate shared OMIS (parallel workers)")
    parser.add_argument("--no_dts",      action="store_true")
    parser.add_argument("--seen_only",   action="store_true")
    parser.add_argument("--unseen_only", action="store_true")
    parser.add_argument("--e_switch",    type=int, default=E_SWITCH)
    parser.add_argument("--n_episodes",  type=int, default=TOTAL_EVAL_EPISODES)
    parser.add_argument("--numWorkers",  type=int, default=1)
    parser.add_argument("--device",      type=str, default="cpu")
    args = parser.parse_args()

    if not os.path.exists(OMIS_CKPT):
        print(f"ERROR: OMIS checkpoint not found: {OMIS_CKPT}")
        print("Run trainOMIS.py first.")
        return

    use_dts = not args.no_dts
    algo    = "OMIS" if use_dts else "OMIS_w_o_Search"
    worldSize = args.numWorkers

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {algo}  |  Workers: {worldSize}  |  Episodes: {args.n_episodes}")
    print(f"{'=' * 60}")
    print(f"{'Ep':>5} {'Return':>8} {'Win%':>8} {'PrAcc':>8} {'PyAcc':>8} {'Srch%':>8} {'EpSec':>6}")
    print("-" * 58)

    # Launch workers
    result_queue = mp.Queue()
    if worldSize > 1:
        procs = []
        for rank in range(worldSize):
            p = mp.Process(target=workerFn, args=(rank, worldSize, args, result_queue))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        workerFn(0, 1, args, result_queue)

    # Collect and log all results
    all_metrics_final = []
    while not result_queue.empty():
        all_metrics_final.extend(result_queue.get())

    if not all_metrics_final:
        print("ERROR: No evaluation data collected. Check Malmo connection.")
        return

    os.makedirs(OMIS_CKPT_DIR, exist_ok=True)
    predatorLogger = MARLLogger(
        algo_name    = f"{algo}_predator",
        log_interval = 10,
        log_file     = os.path.join(OMIS_CKPT_DIR, "omis_predator_metrics.jsonl"),
    )
    preyLogger = MARLLogger(
        algo_name    = f"{algo}_prey",
        log_interval = 10,
        log_file     = os.path.join(OMIS_CKPT_DIR, "omis_prey_metrics.jsonl"),
    )

    for ep_idx, m in enumerate(all_metrics_final):
        predatorLogger.log_episode(episode=ep_idx + 1, episode_return=m["episode_return"],
                                   win=m["win"], opponent_type=m["opp_type"])
        preyLogger.log_episode(episode=ep_idx + 1, episode_return=0.0,
                               win=0 if m["win"] else 1, opponent_type="omis")

    predatorLogger.print_final_summary()
    preyLogger.print_final_summary()

    # Save results JSON
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"eval_results_{algo}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"algorithm": algo, "total_episodes": len(all_metrics),
                   "workers": worldSize}, f, indent=2)
    print(f"\n[evalOMIS] Results saved → {json_path}")


if __name__ == "__main__":
    main()
