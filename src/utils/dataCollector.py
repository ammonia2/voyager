"""
dataCollector.py
================
Collect pretraining data for predator-shared OMIS.

Per episode we record two perspectives (Predator1 and Predator2), each with:
- self action: predator flat action index in [0, 29]
- predator-opponent action: other predator flat action index in [0, 29]
- prey-opponent action: prey flat action index in [0, 8]
"""

import os
import random
import pickle
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.brAgent import BRActorCritic
from utils.obsUtils import flattenObs
from utils.scriptedPolicies import (
    get_prey_policy_by_index,
    encode_prey_action_9,
    decode_pred_action_30,
    turn_bin5_to_cont,
)


K_POLICIES = 10
H_EPI = 15
C_EPI = 3
B_SEQ = 20
GAMMA = 1.0
REWARD_SCALING = 100.0

SELF_AGENT_INDICES = [0, 1]
OPP_INDICES = [2]
PREY_IDX = 2


class EpisodeData:
    """Stores one trajectory from one predator perspective."""

    def __init__(self, policy_idx: int, self_agent_idx: int):
        self.policy_idx = policy_idx
        self.self_agent_idx = self_agent_idx
        self.states = []
        self.self_actions = []
        self.pred_opp_actions = []
        self.prey_opp_actions = []
        self.rewards = []
        self.rtgs = []
        self.length = 0

    def add_step(self, state, self_act, pred_opp_act, prey_opp_act, reward):
        self.states.append(np.array(state, dtype=np.float32))
        self.self_actions.append(int(self_act))
        self.pred_opp_actions.append(int(pred_opp_act))
        self.prey_opp_actions.append(int(prey_opp_act))
        self.rewards.append(float(reward) / REWARD_SCALING)
        self.length += 1

    def compute_rtgs(self):
        self.rtgs = [0.0] * len(self.rewards)
        running = 0.0
        for t in reversed(range(len(self.rewards))):
            running = self.rewards[t] + GAMMA * running
            self.rtgs[t] = running

    def get_epi_tokens(self):
        """Return tokens used to build D_epi."""
        return list(zip(self.states, self.pred_opp_actions, self.prey_opp_actions))


class PretrainSample:
    """Single training sample at timestep t."""

    def __init__(
        self,
        policy_idx,
        timestep,
        state,
        epi_states,
        epi_pred_opp_acts,
        epi_prey_opp_acts,
        step_states,
        step_self_acts,
        step_pred_opp_acts,
        step_prey_opp_acts,
        step_rtgs,
        step_timesteps,
        label_self_act,
        label_pred_opp_act,
        label_prey_opp_act,
        label_rtg,
    ):
        self.policy_idx = policy_idx
        self.timestep = timestep
        self.state = state
        self.epi_states = epi_states
        self.epi_pred_opp_acts = epi_pred_opp_acts
        self.epi_prey_opp_acts = epi_prey_opp_acts
        self.step_states = step_states
        self.step_self_acts = step_self_acts
        self.step_pred_opp_acts = step_pred_opp_acts
        self.step_prey_opp_acts = step_prey_opp_acts
        self.step_rtgs = step_rtgs
        self.step_timesteps = step_timesteps
        self.label_self_act = label_self_act
        self.label_pred_opp_act = label_pred_opp_act
        self.label_prey_opp_act = label_prey_opp_act
        self.label_rtg = label_rtg


def build_d_epi(historical_episodes, h_epi=H_EPI, c_epi=C_EPI):
    """Build D_epi from historical trajectories."""
    seg_len = h_epi // c_epi

    epi_states = []
    epi_pred_acts = []
    epi_prey_acts = []

    sampled = random.sample(historical_episodes, min(c_epi, len(historical_episodes)))
    for ep in sampled:
        tokens = ep.get_epi_tokens()
        if len(tokens) < seg_len:
            tokens = tokens + [tokens[-1]] * (seg_len - len(tokens))
        max_start = max(0, len(tokens) - seg_len)
        start = random.randint(0, max_start)
        segment = tokens[start : start + seg_len]
        for state, pred_act, prey_act in segment:
            epi_states.append(state)
            epi_pred_acts.append(pred_act)
            epi_prey_acts.append(prey_act)

    while len(epi_states) < h_epi:
        epi_states.append(epi_states[-1].copy() if epi_states else np.zeros(128, dtype=np.float32))
        epi_pred_acts.append(epi_pred_acts[-1] if epi_pred_acts else 0)
        epi_prey_acts.append(epi_prey_acts[-1] if epi_prey_acts else 0)

    return (
        np.array(epi_states[:h_epi], dtype=np.float32),
        np.array(epi_pred_acts[:h_epi], dtype=np.int64),
        np.array(epi_prey_acts[:h_epi], dtype=np.int64),
    )


def build_pretrain_samples(episode, historical_episodes, b_seq=B_SEQ):
    """Build per-timestep PretrainSamples for one perspective trajectory."""
    samples = []
    t_len = episode.length
    if t_len == 0:
        return samples

    if len(historical_episodes) >= C_EPI:
        epi_states, epi_pred_acts, epi_prey_acts = build_d_epi(historical_episodes)
    else:
        epi_states = np.zeros((H_EPI, 128), dtype=np.float32)
        epi_pred_acts = np.zeros(H_EPI, dtype=np.int64)
        epi_prey_acts = np.zeros(H_EPI, dtype=np.int64)

    for t in range(t_len):
        start_t = max(0, t - b_seq)

        window_states = episode.states[start_t:t]
        window_self_acts = episode.self_actions[start_t:t]
        window_pred_opp_acts = episode.pred_opp_actions[start_t:t]
        window_prey_opp_acts = episode.prey_opp_actions[start_t:t]
        window_rtgs = episode.rtgs[start_t:t]
        window_timesteps = list(range(start_t, t))

        step_states = window_states + [episode.states[t]]
        step_timesteps = window_timesteps + [t]

        sample = PretrainSample(
            policy_idx=episode.policy_idx,
            timestep=t,
            state=episode.states[t],
            epi_states=epi_states,
            epi_pred_opp_acts=epi_pred_acts,
            epi_prey_opp_acts=epi_prey_acts,
            step_states=np.array(step_states, dtype=np.float32),
            step_self_acts=np.array(window_self_acts, dtype=np.int64),
            step_pred_opp_acts=np.array(window_pred_opp_acts, dtype=np.int64),
            step_prey_opp_acts=np.array(window_prey_opp_acts, dtype=np.int64),
            step_rtgs=np.array(window_rtgs, dtype=np.float32),
            step_timesteps=np.array(step_timesteps, dtype=np.int64),
            label_self_act=episode.self_actions[t],
            label_pred_opp_act=episode.pred_opp_actions[t],
            label_prey_opp_act=episode.prey_opp_actions[t],
            label_rtg=episode.rtgs[t],
        )
        samples.append(sample)

    return samples


class DataCollector:
    """Collect OMIS pretraining data from both predator perspectives."""

    def __init__(self, env, br_checkpoints_dir, save_dir, device="cpu"):
        self.env = env
        self.br_ckpt_dir = br_checkpoints_dir
        self.save_dir = save_dir
        self.device = device
        os.makedirs(save_dir, exist_ok=True)

    def collect(self, n_episodes_per_policy=500, verbose=True):
        total_samples = 0

        for k in range(K_POLICIES):
            br_path = os.path.join(self.br_ckpt_dir, f"br_policy_{k}.pt")
            if not os.path.exists(br_path):
                print(f"[DataCollector] Warning: BR checkpoint missing for policy {k}: {br_path}")
                continue

            br_net = BRActorCritic().to(self.device)
            ckpt = __import__("torch").load(br_path, map_location=self.device)
            br_net.load_state_dict(ckpt["net"])
            br_net.eval()

            prey_policy = get_prey_policy_by_index(k)
            historical_eps = []
            all_samples = []

            if verbose:
                print(f"\n[DataCollector] Collecting for prey policy {k} ({prey_policy.name})")

            for ep_idx in range(n_episodes_per_policy):
                ep_pred0, ep_pred1 = self._run_episode(br_net, prey_policy, k)

                for ep_data in (ep_pred0, ep_pred1):
                    ep_data.compute_rtgs()
                    samples = build_pretrain_samples(ep_data, historical_eps)
                    all_samples.extend(samples)
                    historical_eps.append(ep_data)

                if verbose and (ep_idx + 1) % 50 == 0:
                    ep_ret = sum(ep_pred0.rewards) + sum(ep_pred1.rewards)
                    print(
                        f"  Policy {k:2d} | Episode {ep_idx+1:4d}/{n_episodes_per_policy} "
                        f"| Combined Return {ep_ret:.2f} | Samples {len(all_samples)}"
                    )

            save_path = os.path.join(self.save_dir, f"pretrain_data_policy_{k}.pkl")
            with open(save_path, "wb") as file:
                pickle.dump(all_samples, file)

            total_samples += len(all_samples)
            if verbose:
                print(f"[DataCollector] Policy {k} done -> {len(all_samples)} samples saved to {save_path}")

        if verbose:
            print(f"\n[DataCollector] Total samples collected: {total_samples}")
        return total_samples

    def _run_episode(self, br_net, prey_policy, policy_idx):
        """Run one episode and return two EpisodeData objects (predator-0 and predator-1 views)."""
        import torch

        ep_pred0 = EpisodeData(policy_idx, self_agent_idx=0)
        ep_pred1 = EpisodeData(policy_idx, self_agent_idx=1)

        obs_all = self.env.reset()
        done = False
        max_steps = getattr(self.env, "EPISODE_STEP_LIMIT", 500)

        last_actions = [
            (2, 0.0, 1),
            (2, 0.0, 1),
            (2, 2, 0),
        ]

        for _ in range(max_steps):
            if done:
                break

            state_pred0 = self._encode_obs(obs_all, last_actions, agent_idx=0)
            state_pred1 = self._encode_obs(obs_all, last_actions, agent_idx=1)

            act0_flat = br_net.act(state_pred0)
            act1_flat = br_net.act(state_pred1)

            move0, turn_bin0, attack0 = decode_pred_action_30(act0_flat)
            move1, turn_bin1, attack1 = decode_pred_action_30(act1_flat)

            prey_move, prey_turn = prey_policy(obs_all[PREY_IDX], PREY_IDX)
            prey_act_flat = encode_prey_action_9(prey_move, prey_turn)

            actions = [
                (move0, turn_bin5_to_cont(turn_bin0), attack0),
                (move1, turn_bin5_to_cont(turn_bin1), attack1),
                (prey_move, prey_turn, 0),
            ]

            obs_next, rewards, dones = self.env.step(actions)
            done = bool(dones[0])

            ep_pred0.add_step(
                state=state_pred0,
                self_act=act0_flat,
                pred_opp_act=act1_flat,
                prey_opp_act=prey_act_flat,
                reward=rewards[0],
            )
            ep_pred1.add_step(
                state=state_pred1,
                self_act=act1_flat,
                pred_opp_act=act0_flat,
                prey_opp_act=prey_act_flat,
                reward=rewards[1],
            )

            obs_all = obs_next
            last_actions = actions

        return ep_pred0, ep_pred1

    def _encode_obs(self, obs_all: list[dict], last_actions: list[tuple], agent_idx: int):
        """Encode one predator observation using obsUtils flattening + VoxelEncoder."""
        import torch
        from models.voxelEncoder import VoxelEncoder

        if not hasattr(self, "_voxel_encoder"):
            self._voxel_encoder = VoxelEncoder().to(self.device)
            self._voxel_encoder.eval()

        flat = flattenObs(agent_idx, obs_all[agent_idx], obs_all, last_actions)
        flat_t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            state = self._voxel_encoder(flat_t)
        return state.squeeze(0).cpu().numpy()


class PretrainDataset:
    """Dataset wrapper around collected PretrainSamples."""

    def __init__(self, data_dir, policy_indices=None):
        if policy_indices is None:
            policy_indices = list(range(K_POLICIES))

        self.samples = []
        for k in policy_indices:
            path = os.path.join(data_dir, f"pretrain_data_policy_{k}.pkl")
            if not os.path.exists(path):
                print(f"[Dataset] Missing data for policy {k}: {path}")
                continue
            with open(path, "rb") as file:
                loaded = pickle.load(file)
            self.samples.extend(loaded)
            print(f"[Dataset] Loaded {len(loaded)} samples for policy {k}")

        print(f"[Dataset] Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch

        sample = self.samples[idx]
        return {
            "epi_states": torch.tensor(sample.epi_states, dtype=torch.float32),
            "epi_pred_opp_acts": torch.tensor(sample.epi_pred_opp_acts, dtype=torch.long),
            "epi_prey_opp_acts": torch.tensor(sample.epi_prey_opp_acts, dtype=torch.long),
            "step_states": torch.tensor(sample.step_states, dtype=torch.float32),
            "step_self_acts": torch.tensor(sample.step_self_acts, dtype=torch.long),
            "step_pred_opp_acts": torch.tensor(sample.step_pred_opp_acts, dtype=torch.long),
            "step_prey_opp_acts": torch.tensor(sample.step_prey_opp_acts, dtype=torch.long),
            "step_rtgs": torch.tensor(sample.step_rtgs, dtype=torch.float32),
            "step_timesteps": torch.tensor(sample.step_timesteps, dtype=torch.long),
            "label_self_act": torch.tensor(sample.label_self_act, dtype=torch.long),
            "label_pred_opp_act": torch.tensor(sample.label_pred_opp_act, dtype=torch.long),
            "label_prey_opp_act": torch.tensor(sample.label_prey_opp_act, dtype=torch.long),
            "label_rtg": torch.tensor(sample.label_rtg, dtype=torch.float32),
        }

    @staticmethod
    def collate_fn(batch):
        import torch
        from torch.nn.utils.rnn import pad_sequence

        fixed_keys = [
            "epi_states",
            "epi_pred_opp_acts",
            "epi_prey_opp_acts",
            "label_self_act",
            "label_pred_opp_act",
            "label_prey_opp_act",
            "label_rtg",
        ]
        variable_keys = [
            "step_states",
            "step_self_acts",
            "step_pred_opp_acts",
            "step_prey_opp_acts",
            "step_rtgs",
            "step_timesteps",
        ]

        out = {}
        for key in fixed_keys:
            out[key] = torch.stack([item[key] for item in batch], dim=0)

        for key in variable_keys:
            sequences = [item[key] for item in batch]
            out[key] = pad_sequence(sequences, batch_first=True, padding_value=0)

        return out
