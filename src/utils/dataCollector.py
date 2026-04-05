"""
dataCollector.py
================
Collects pretraining data for the OMIS Transformer.

Two phases:
  Phase 1 — BR data collection:
    For each scripted policy π⁻¹,k in Pi_train:
      Run the BR agent (prey) against the fixed scripted predators
      Collect (state, self_action, opp_joint_action, reward, next_state) tuples
      Also collect full episode trajectories for D_epi construction

  Phase 2 — Pretraining dataset construction:
    For each timestep t in each episode:
      Build D_t^k = (s_t, D_t^k, a^{1,k,*}_t, a^{-1,k}_t, G^{1,k,*}_t)
      per Equation (2) of the paper.

D_epi construction (Appendix C):
  Sample C=3 trajectories from all historical games involving π⁻¹,k
  From each: sample consecutive segment of length H/C = 5 steps: (s_h, ã⁻¹_h)
  Concatenate → H=15 token pairs

D_step construction (Section 4.1):
  Current episode's running history: (s0, a⁻¹0, ..., st-1, a⁻¹t-1)

RTG computation:
  G^{1,k,*}_t = Σ_{t'=t}^{T} γ^{t'-t} r^1_{t'}
  with γ=1.0 (Appendix H.2) and reward_scaling=100

Data is stored as a list of episode dicts and saved to disk.
"""

import os
import math
import random
import pickle
import numpy as np
import sys

# Add src to path so we can import from sibling modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.brAgent import BRActorCritic
from utils.scriptedPolicies import get_policy_by_index, get_flat_action


# ─────────────────────────────────────────────────────────────────
# Constants (from Appendix H.2)
# ─────────────────────────────────────────────────────────────────

K_POLICIES      = 10      # Number of scripted policies in Pi_train
H_EPI           = 15      # D_epi sequence length
C_EPI           = 3       # Trajectories sampled to build D_epi
B_SEQ           = 20      # Max step window for Transformer
GAMMA           = 1.0     # Discount for RTG
REWARD_SCALING  = 100.0   # Reward scaling factor (Appendix H.2)
SELF_AGENT_IDX  = 2       # Prey is agent index 2
OPP_INDICES     = [0, 1]  # Predator1, Predator2
N_OPP_ACTIONS   = 18      # Action space per opponent


# ─────────────────────────────────────────────────────────────────
# Episode data structure
# ─────────────────────────────────────────────────────────────────

class EpisodeData:
    """
    Stores all data for one episode played by BR against policy k.

    Attributes:
      policy_idx    : int — which scripted policy (k)
      states        : list of np.array(128,) — VoxelEncoder outputs per step
      self_actions  : list of int — flat action indices for the prey
      opp_actions   : list of int — JOINT flat action index for both predators
      rewards       : list of float — scaled rewards for the prey
      rtgs          : list of float — return-to-go from each timestep
      length        : int — number of steps
    """

    def __init__(self, policy_idx):
        self.policy_idx   = policy_idx
        self.states       = []
        self.self_actions = []
        self.opp_actions  = []   # joint action index = act0 * 18 + act1
        self.rewards      = []
        self.rtgs         = []
        self.length       = 0

    def add_step(self, state, self_act, opp_act0, opp_act1, reward):
        self.states.append(np.array(state, dtype=np.float32))
        self.self_actions.append(int(self_act))
        self.opp_actions.append(int(opp_act0 * N_OPP_ACTIONS + opp_act1))
        self.rewards.append(float(reward) / REWARD_SCALING)
        self.length += 1

    def compute_rtgs(self):
        """Compute return-to-go for each timestep (γ=1.0)."""
        T = len(self.rewards)
        self.rtgs = [0.0] * T
        running   = 0.0
        for t in reversed(range(T)):
            running       = self.rewards[t] + GAMMA * running
            self.rtgs[t]  = running

    def get_epi_tokens(self):
        """
        Return (states, opp_actions) suitable for D_epi construction.
        Returns list of (state_array, opp_joint_action_int) tuples.
        """
        return list(zip(self.states, self.opp_actions))


# ─────────────────────────────────────────────────────────────────
# Pretraining sample
# ─────────────────────────────────────────────────────────────────

class PretrainSample:
    """
    One pretraining sample D_t^k for timestep t of episode with policy k.
    Corresponds to Equation (2): D_t^k = (s_t, D_t^k, a^{1,k,*}_t, a^{-1,k}_t, G^{1,k,*}_t)

    Fields:
      policy_idx      : int k
      timestep        : int t
      state           : np.array(128,)    — s_t
      epi_states      : np.array(H, 128)  — D_epi states
      epi_opp_acts    : np.array(H,) int  — D_epi joint opp actions
      step_states     : np.array(L, 128)  — D_step states (window ending at s_t)
      step_self_acts  : np.array(L-1,) int
      step_opp_acts   : np.array(L-1,) int
      step_rtgs       : np.array(L-1,)    — RTGs for D_step
      step_timesteps  : np.array(L,) int
      label_self_act  : int               — a^{1,k,*}_t  (actor label)
      label_opp_act   : int               — a^{-1,k}_t   (imitator label)
      label_rtg       : float             — G^{1,k,*}_t  (critic label)
    """

    def __init__(
        self, policy_idx, timestep,
        state,
        epi_states, epi_opp_acts,
        step_states, step_self_acts, step_opp_acts, step_rtgs, step_timesteps,
        label_self_act, label_opp_act, label_rtg,
    ):
        self.policy_idx     = policy_idx
        self.timestep       = timestep
        self.state          = state
        self.epi_states     = epi_states
        self.epi_opp_acts   = epi_opp_acts
        self.step_states    = step_states
        self.step_self_acts = step_self_acts
        self.step_opp_acts  = step_opp_acts
        self.step_rtgs      = step_rtgs
        self.step_timesteps = step_timesteps
        self.label_self_act = label_self_act
        self.label_opp_act  = label_opp_act
        self.label_rtg      = label_rtg


# ─────────────────────────────────────────────────────────────────
# D_epi builder (Appendix C)
# ─────────────────────────────────────────────────────────────────

def build_d_epi(historical_episodes, h_epi=H_EPI, c_epi=C_EPI):
    """
    Build D_epi from C historical episodes involving the same policy.

    historical_episodes: list of EpisodeData objects
    Returns:
      epi_states   : np.array(H, 128)
      epi_opp_acts : np.array(H,) int
    """
    seg_len = h_epi // c_epi   # 15 // 3 = 5 steps per trajectory

    epi_states   = []
    epi_opp_acts = []

    # Sample C trajectories (or fewer if not enough available)
    sampled = random.sample(historical_episodes, min(c_epi, len(historical_episodes)))

    for ep in sampled:
        tokens = ep.get_epi_tokens()   # list of (state, opp_act) tuples
        if len(tokens) < seg_len:
            tokens = tokens + [tokens[-1]] * (seg_len - len(tokens))
        max_start = max(0, len(tokens) - seg_len)
        start     = random.randint(0, max_start)
        segment   = tokens[start: start + seg_len]
        for s, a in segment:
            epi_states.append(s)
            epi_opp_acts.append(a)

    # Pad to exactly H if fewer than C trajectories were available
    while len(epi_states) < h_epi:
        epi_states.append(epi_states[-1].copy() if epi_states else np.zeros(128, dtype=np.float32))
        epi_opp_acts.append(epi_opp_acts[-1] if epi_opp_acts else 0)

    epi_states   = np.array(epi_states[:h_epi],   dtype=np.float32)
    epi_opp_acts = np.array(epi_opp_acts[:h_epi], dtype=np.int64)
    return epi_states, epi_opp_acts


# ─────────────────────────────────────────────────────────────────
# Pretraining dataset builder
# ─────────────────────────────────────────────────────────────────

def build_pretrain_samples(episode, historical_episodes, b_seq=B_SEQ):
    """
    Build PretrainSample objects for every timestep in an episode.
    Follows Algorithm 1 lines 7-8 and Section 4.1.

    episode             : EpisodeData (current episode)
    historical_episodes : list of EpisodeData (past episodes for this policy)
    Returns list of PretrainSample objects.
    """
    samples = []
    T = episode.length

    if T == 0:
        return samples

    # Build D_epi from historical trajectories
    if len(historical_episodes) >= C_EPI:
        epi_states, epi_opp_acts = build_d_epi(historical_episodes)
    else:
        # Not enough history yet — use zeros
        epi_states   = np.zeros((H_EPI, 128), dtype=np.float32)
        epi_opp_acts = np.zeros(H_EPI, dtype=np.int64)

    # For each timestep t, build D_step and collect labels
    for t in range(T):
        # D_step window: last b_seq steps of current episode up to (but not including) t
        start_t = max(0, t - b_seq)
        window_states    = episode.states[start_t: t]        # (t - start_t, 128)
        window_self_acts = episode.self_actions[start_t: t]  # (t - start_t,)
        window_opp_acts  = episode.opp_actions[start_t: t]
        window_rtgs      = episode.rtgs[start_t: t]
        window_timesteps = list(range(start_t, t))

        # Add the query state s_t at the end
        step_states    = window_states + [episode.states[t]]
        step_timesteps = window_timesteps + [t]
        L = len(step_states)

        step_states_arr    = np.array(step_states,    dtype=np.float32)   # (L, 128)
        step_self_acts_arr = np.array(window_self_acts, dtype=np.int64)   # (L-1,)
        step_opp_acts_arr  = np.array(window_opp_acts,  dtype=np.int64)   # (L-1,)
        step_rtgs_arr      = np.array(window_rtgs,     dtype=np.float32)  # (L-1,)
        step_timesteps_arr = np.array(step_timesteps,  dtype=np.int64)    # (L,)

        # Labels (Equations 3, 4, 5)
        label_self_act = episode.self_actions[t]   # a^{1,k,*}_t
        label_opp_act  = episode.opp_actions[t]    # a^{-1,k}_t
        label_rtg      = episode.rtgs[t]            # G^{1,k,*}_t

        sample = PretrainSample(
            policy_idx     = episode.policy_idx,
            timestep       = t,
            state          = episode.states[t],
            epi_states     = epi_states,
            epi_opp_acts   = epi_opp_acts,
            step_states    = step_states_arr,
            step_self_acts = step_self_acts_arr,
            step_opp_acts  = step_opp_acts_arr,
            step_rtgs      = step_rtgs_arr,
            step_timesteps = step_timesteps_arr,
            label_self_act = label_self_act,
            label_opp_act  = label_opp_act,
            label_rtg      = label_rtg,
        )
        samples.append(sample)

    return samples


# ─────────────────────────────────────────────────────────────────
# DataCollector — runs collection loop
# ─────────────────────────────────────────────────────────────────

class DataCollector:
    """
    Collects pretraining data by running the trained BR agent against
    each scripted policy in Pi_train.

    Usage:
        collector = DataCollector(env, br_checkpoints_dir, save_dir)
        collector.collect(n_episodes_per_policy=500)
    """

    def __init__(self, env, br_checkpoints_dir, save_dir, device="cpu"):
        """
        env                  : MalmoEnv instance
        br_checkpoints_dir   : path to directory with saved BR checkpoints
        save_dir             : where to save collected pretraining data
        """
        self.env              = env
        self.br_ckpt_dir      = br_checkpoints_dir
        self.save_dir         = save_dir
        self.device           = device
        os.makedirs(save_dir, exist_ok=True)

    def collect(self, n_episodes_per_policy=500, verbose=True):
        """
        Collect pretraining data for all K policies.

        For each policy k:
          1. Load trained BR_k network
          2. Run n_episodes episodes with BR_k as prey, policy k as predators
          3. Build PretrainSamples for each episode
          4. Save to disk

        Returns: total number of PretrainSamples collected
        """
        total_samples = 0

        for k in range(K_POLICIES):
            br_path = os.path.join(self.br_ckpt_dir, f"br_policy_{k}.pt")
            if not os.path.exists(br_path):
                print(f"[DataCollector] Warning: BR checkpoint not found for policy {k}: {br_path}")
                print(f"[DataCollector] Skipping policy {k}. Run trainBR.py first.")
                continue

            # Load BR network
            br_net = BRActorCritic().to(self.device)
            ckpt   = __import__("torch").load(br_path, map_location=self.device)
            br_net.load_state_dict(ckpt["net"])
            br_net.eval()

            # Load scripted policy
            scripted_policy = get_policy_by_index(k)

            # Collection loop
            historical_eps = []
            all_samples    = []

            if verbose:
                print(f"\n[DataCollector] Collecting for policy {k} ({scripted_policy.name})")

            for ep_idx in range(n_episodes_per_policy):
                ep_data = self._run_episode(br_net, scripted_policy, k)
                ep_data.compute_rtgs()

                # Build pretraining samples for this episode
                samples = build_pretrain_samples(ep_data, historical_eps)
                all_samples.extend(samples)
                historical_eps.append(ep_data)

                if verbose and (ep_idx + 1) % 50 == 0:
                    ep_ret = sum(ep_data.rewards)
                    print(
                        f"  Policy {k:2d} | Episode {ep_idx+1:4d}/{n_episodes_per_policy} "
                        f"| Return {ep_ret:.2f} | Samples {len(all_samples)}"
                    )

            # Save samples for this policy
            save_path = os.path.join(self.save_dir, f"pretrain_data_policy_{k}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(all_samples, f)

            total_samples += len(all_samples)
            if verbose:
                print(f"[DataCollector] Policy {k} done → {len(all_samples)} samples saved to {save_path}")

        if verbose:
            print(f"\n[DataCollector] Total samples collected: {total_samples}")

        return total_samples

    def _run_episode(self, br_net, scripted_policy, policy_idx):
        """
        Run one episode.
        br_net          : BRActorCritic (prey agent)
        scripted_policy : callable predator policy
        policy_idx      : int k
        Returns EpisodeData
        """
        from models.voxelEncoder import VoxelEncoder
        import torch

        ep_data = EpisodeData(policy_idx)
        obs_all = self.env.reset()

        # We need a VoxelEncoder to convert raw obs to state vectors
        # (re-use the one from the env if available, else create a new one)
        if not hasattr(self, "_voxel_encoder"):
            self._voxel_encoder = VoxelEncoder().to(self.device)
            self._voxel_encoder.eval()

        done     = False
        max_steps = getattr(self.env, "episodeStepLimit", 500)

        for t in range(max_steps):
            if done:
                break

            # Encode prey observation → state vector
            prey_obs  = obs_all[SELF_AGENT_IDX]
            prey_state = self._encode_obs(prey_obs)

            # BR agent selects action for prey
            self_act = br_net.act(prey_state)

            # Scripted policy selects actions for predators
            opp_act0 = get_flat_action(
                scripted_policy(obs_all[OPP_INDICES[0]], OPP_INDICES[0])
            )
            opp_act1 = get_flat_action(
                scripted_policy(obs_all[OPP_INDICES[1]], OPP_INDICES[1])
            )

            # Build joint action list for env.step()
            # env expects [action_agent0, action_agent1, action_agent2]
            actions = [opp_act0, opp_act1, self_act]

            obs_next, rewards, dones, _ = self.env.step(actions)
            prey_reward = rewards[SELF_AGENT_IDX]
            done        = dones[SELF_AGENT_IDX]

            ep_data.add_step(prey_state, self_act, opp_act0, opp_act1, prey_reward)
            obs_all = obs_next

        return ep_data

    def _encode_obs(self, obs):
        """
        Convert raw Malmo observation dict to 128-dim state vector.
        obs: dict with 'ob', 'entities', 'x', 'z', 'yaw', 'life'
        """
        import torch
        import numpy as np

        if not hasattr(self, "_voxel_encoder"):
            from models.voxelEncoder import VoxelEncoder
            self._voxel_encoder = VoxelEncoder().to(self.device)
            self._voxel_encoder.eval()

        # Build input tensors for VoxelEncoder
        voxel_grid = torch.tensor(
            obs.get("ob", [0] * 49), dtype=torch.long
        ).unsqueeze(0).to(self.device)

        # Entity tensor: (1, maxEntities, 5)
        entities_raw = obs.get("entities", [])
        max_ents     = 4
        ent_tensor   = torch.zeros(1, max_ents, 5, device=self.device)
        ent_mask     = torch.zeros(1, max_ents, device=self.device)
        for i, e in enumerate(entities_raw[:max_ents]):
            ent_tensor[0, i] = torch.tensor(e[:5], dtype=torch.float32)
            ent_mask[0, i]   = 1.0

        # Stats tensor: (1, 4) = [x, z, yaw, life]
        stats = torch.tensor(
            [obs.get("x", 0.0), obs.get("z", 0.0),
             obs.get("yaw", 0.0), obs.get("life", 20.0)],
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            state = self._voxel_encoder(voxel_grid, ent_tensor, ent_mask, stats)

        return state.squeeze(0).cpu().numpy()   # (128,)


# ─────────────────────────────────────────────────────────────────
# Dataset loader for training
# ─────────────────────────────────────────────────────────────────

class PretrainDataset:
    """
    PyTorch-compatible dataset wrapping collected PretrainSamples.
    Used by trainOMIS.py.
    """

    def __init__(self, data_dir, policy_indices=None):
        """
        data_dir       : directory containing pretrain_data_policy_k.pkl files
        policy_indices : list of k values to load (default: all K_POLICIES)
        """
        import pickle
        if policy_indices is None:
            policy_indices = list(range(K_POLICIES))

        self.samples = []
        for k in policy_indices:
            path = os.path.join(data_dir, f"pretrain_data_policy_{k}.pkl")
            if not os.path.exists(path):
                print(f"[Dataset] Missing data for policy {k}: {path}")
                continue
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            self.samples.extend(loaded)
            print(f"[Dataset] Loaded {len(loaded)} samples for policy {k}")

        print(f"[Dataset] Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        import torch
        return {
            "epi_states":      torch.tensor(s.epi_states,     dtype=torch.float32),
            "epi_opp_acts":    torch.tensor(s.epi_opp_acts,   dtype=torch.long),
            "step_states":     torch.tensor(s.step_states,    dtype=torch.float32),
            "step_self_acts":  torch.tensor(s.step_self_acts, dtype=torch.long),
            "step_opp_acts":   torch.tensor(s.step_opp_acts,  dtype=torch.long),
            "step_rtgs":       torch.tensor(s.step_rtgs,      dtype=torch.float32),
            "step_timesteps":  torch.tensor(s.step_timesteps, dtype=torch.long),
            "label_self_act":  torch.tensor(s.label_self_act, dtype=torch.long),
            "label_opp_act":   torch.tensor(s.label_opp_act,  dtype=torch.long),
            "label_rtg":       torch.tensor(s.label_rtg,      dtype=torch.float32),
        }

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate: pads variable-length step sequences to the same length.
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence

        keys_fixed = ["epi_states", "epi_opp_acts", "label_self_act", "label_opp_act", "label_rtg"]
        keys_var   = ["step_states", "step_self_acts", "step_opp_acts", "step_rtgs", "step_timesteps"]

        out = {}

        # Fixed-length fields: just stack
        for k in keys_fixed:
            out[k] = torch.stack([item[k] for item in batch], dim=0)

        # Variable-length fields: pad to max length in batch
        for k in keys_var:
            seqs = [item[k] for item in batch]
            # pad_sequence expects (T, ...) tensors → we need (B, T_max, ...)
            padded = pad_sequence(seqs, batch_first=True, padding_value=0)
            out[k] = padded

        return out
