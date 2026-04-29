"""
omisAgent.py
============
Shared OMIS Transformer for dual-predator learning.

Design:
- One shared model instance for both predators
- Self action space (predator): 3 move * 5 turn bins * 2 attack = 30
- Opponent modeling is split into two branches:
  - Other predator: 30 classes
  - Prey: 9 classes (3 move * 3 turn)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_DIM = 147

TURN_BINS = 5
N_SELF_ACTIONS = 30
N_PRED_OPP_ACTIONS = 30
N_PREY_ACTIONS = 9

D_MODEL = 32
D_FF = 128
N_HEADS = 1
N_BLOCKS = 3
DROPOUT = 0.1

H_EPI = 15
C_EPI = 3
B_SEQ = 20

TOKEN_STATE = 0
TOKEN_SELF_ACT = 1
TOKEN_OPP_ACT = 2
TOKEN_RTG = 3
TOKEN_EPI_STATE = 4
TOKEN_EPI_ACT = 5
N_TOKEN_TYPES = 6


# ---------------------------------------------------------------------------
# Turn discretization utilities
# ---------------------------------------------------------------------------


def discretize_turn(turn_cont: float) -> int:
    """Map continuous turn in [-1, 1] to bin index [0, TURN_BINS-1]."""
    clipped = max(-1.0, min(1.0, float(turn_cont)))
    return int((clipped + 1.0) / 2.0 * (TURN_BINS - 1) + 0.5)


def undiscretize_turn(bin_idx: int) -> float:
    """Map turn bin index [0, TURN_BINS-1] to continuous turn in [-1, 1]."""
    idx = max(0, min(TURN_BINS - 1, int(bin_idx)))
    return (idx / (TURN_BINS - 1)) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Action encoding utilities
# ---------------------------------------------------------------------------


def encode_self_action(move: int, turn_bin: int, attack: int) -> int:
    """Encode predator action to flat index [0, 29]."""
    return int(move) * (TURN_BINS * 2) + int(turn_bin) * 2 + int(attack)


def decode_self_action(idx: int) -> tuple[int, int, int]:
    """Decode predator action index [0, 29] to (move, turn_bin, attack)."""
    action = int(idx)
    attack = action % 2
    turn_bin = (action // 2) % TURN_BINS
    move = action // (TURN_BINS * 2)
    return move, turn_bin, attack


def encode_pred_opp_action(move: int, turn_bin: int, attack: int) -> int:
    """Encode other predator action to flat index [0, 29]."""
    return encode_self_action(move, turn_bin, attack)


def decode_pred_opp_action(idx: int) -> tuple[int, int, int]:
    """Decode other predator action index [0, 29]."""
    return decode_self_action(idx)


def encode_prey_action(move: int, turn: int) -> int:
    """Encode prey action to flat index [0, 8]."""
    return int(move) * 3 + int(turn)


def decode_prey_action(idx: int) -> tuple[int, int]:
    """Decode prey action index [0, 8] to (move, turn)."""
    action = int(idx)
    turn = action % 3
    move = action // 3
    return move, turn


def action_list_to_malmo(flat_idx: int) -> tuple[int, float, int]:
    """Convert predator flat action index [0, 29] to env tuple format."""
    move, turn_bin, attack = decode_self_action(flat_idx)
    return move, undiscretize_turn(turn_bin), attack


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


def get_timestep_encoding(timesteps: torch.Tensor, d_model: int) -> torch.Tensor:
    """Sinusoidal timestep encoding."""
    _, _ = timesteps.shape
    half = d_model // 2
    div = torch.exp(
        torch.arange(half, dtype=torch.float32, device=timesteps.device)
        * -(math.log(10000.0) / half)
    )
    ts = timesteps.float().unsqueeze(-1)
    sin_enc = torch.sin(ts * div)
    cos_enc = torch.cos(ts * div)
    return torch.cat([sin_enc, cos_enc], dim=-1)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.res_drop = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t_len, channels = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(channels)
        scores = scores.masked_fill(self.causal_mask[:t_len, :t_len].unsqueeze(0), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = torch.bmm(weights, v)
        return self.res_drop(self.out_proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# OMIS model
# ---------------------------------------------------------------------------


class OMISModel(nn.Module):
    """Shared OMIS model with split opponent heads (predator and prey)."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_self_actions: int = N_SELF_ACTIONS,
        d_model: int = D_MODEL,
        d_ff: int = D_FF,
        n_heads: int = N_HEADS,
        n_blocks: int = N_BLOCKS,
        dropout: float = DROPOUT,
        h_epi: int = H_EPI,
        b_seq: int = B_SEQ,
        max_len: int = 512,
    ):
        super().__init__()
        _ = n_heads  # kept for interface consistency

        self.state_dim = state_dim
        self.n_self_actions = n_self_actions
        self.d_model = d_model
        self.h_epi = h_epi
        self.b_seq = b_seq

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LeakyReLU(),
        )
        self.self_act_encoder = nn.Embedding(n_self_actions, d_model)

        self.pred_opp_encoder = nn.Embedding(N_PRED_OPP_ACTIONS, d_model)
        self.prey_opp_encoder = nn.Embedding(N_PREY_ACTIONS, d_model)
        self.opp_fuse = nn.Linear(2 * d_model, d_model)

        self.rtg_encoder = nn.Linear(1, d_model)
        self.token_type_emb = nn.Embedding(N_TOKEN_TYPES, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, dropout, max_len)
            for _ in range(n_blocks)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        self.actor_head = nn.Linear(d_model, n_self_actions)
        self.imitator_pred_head = nn.Linear(d_model, N_PRED_OPP_ACTIONS)
        self.imitator_prey_head = nn.Linear(d_model, N_PREY_ACTIONS)
        self.critic_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _fuse_opp_emb(self, pred_idx: torch.Tensor, prey_idx: torch.Tensor) -> torch.Tensor:
        pred_emb = self.pred_opp_encoder(pred_idx)
        prey_emb = self.prey_opp_encoder(prey_idx)
        return self.opp_fuse(torch.cat([pred_emb, prey_emb], dim=-1))

    def _build_sequence(
        self,
        epi_states: torch.Tensor,
        epi_pred_opp_acts: torch.Tensor,
        epi_prey_opp_acts: torch.Tensor,
        step_states: torch.Tensor,
        step_self_acts: torch.Tensor,
        step_pred_opp_acts: torch.Tensor,
        step_prey_opp_acts: torch.Tensor,
        step_rtgs: torch.Tensor,
        step_timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = epi_states.size(0)
        h_len = epi_states.size(1)
        l_len = step_states.size(1)
        device = epi_states.device

        token_list = []
        ttype_list = []
        tstep_list = []

        for h_idx in range(h_len):
            s_emb = self.state_encoder(epi_states[:, h_idx, :])
            oa_emb = self._fuse_opp_emb(epi_pred_opp_acts[:, h_idx], epi_prey_opp_acts[:, h_idx])
            token_list.extend([s_emb, oa_emb])
            ttype_list.extend([
                torch.full((batch_size,), TOKEN_EPI_STATE, dtype=torch.long, device=device),
                torch.full((batch_size,), TOKEN_EPI_ACT, dtype=torch.long, device=device),
            ])
            tstep_list.extend([
                torch.zeros(batch_size, dtype=torch.long, device=device),
                torch.zeros(batch_size, dtype=torch.long, device=device),
            ])

        for step_idx in range(l_len - 1):
            s_emb = self.state_encoder(step_states[:, step_idx, :])
            sa_emb = self.self_act_encoder(step_self_acts[:, step_idx])
            oa_emb = self._fuse_opp_emb(step_pred_opp_acts[:, step_idx], step_prey_opp_acts[:, step_idx])
            rtg_emb = self.rtg_encoder(step_rtgs[:, step_idx, :])
            token_list.extend([s_emb, sa_emb, oa_emb, rtg_emb])
            ttype_list.extend([
                torch.full((batch_size,), TOKEN_STATE, dtype=torch.long, device=device),
                torch.full((batch_size,), TOKEN_SELF_ACT, dtype=torch.long, device=device),
                torch.full((batch_size,), TOKEN_OPP_ACT, dtype=torch.long, device=device),
                torch.full((batch_size,), TOKEN_RTG, dtype=torch.long, device=device),
            ])
            t = step_timesteps[:, step_idx]
            tstep_list.extend([t, t, t, t])

        q_state = self.state_encoder(step_states[:, l_len - 1, :])
        token_list.append(q_state)
        ttype_list.append(torch.full((batch_size,), TOKEN_STATE, dtype=torch.long, device=device))
        tstep_list.append(step_timesteps[:, l_len - 1])

        tokens = torch.stack(token_list, dim=1)
        ttypes = torch.stack(ttype_list, dim=1)
        tsteps = torch.stack(tstep_list, dim=1)
        return tokens, ttypes, tsteps

    def _encode(self, tokens: torch.Tensor, ttypes: torch.Tensor, tsteps: torch.Tensor) -> torch.Tensor:
        ttype_enc = self.token_type_emb(ttypes)
        tstep_enc = get_timestep_encoding(tsteps, self.d_model)
        x = tokens + ttype_enc + tstep_enc
        for block in self.blocks:
            x = block(x)
        return self.ln_final(x)

    def forward(
        self,
        epi_states: torch.Tensor,
        epi_pred_opp_acts: torch.Tensor,
        epi_prey_opp_acts: torch.Tensor,
        step_states: torch.Tensor,
        step_self_acts: torch.Tensor,
        step_pred_opp_acts: torch.Tensor,
        step_prey_opp_acts: torch.Tensor,
        step_rtgs: torch.Tensor,
        step_timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, ttypes, tsteps = self._build_sequence(
            epi_states,
            epi_pred_opp_acts,
            epi_prey_opp_acts,
            step_states,
            step_self_acts,
            step_pred_opp_acts,
            step_prey_opp_acts,
            step_rtgs,
            step_timesteps,
        )
        h = self._encode(tokens, ttypes, tsteps)

        h_len = epi_states.size(1)
        l_len = step_states.size(1)
        epi_state_positions = list(range(0, 2 * h_len, 2))
        step_state_positions = [2 * h_len + 4 * idx for idx in range(l_len)]
        state_positions = epi_state_positions + step_state_positions

        state_h = h[:, state_positions, :]
        actor_logits = self.actor_head(state_h)
        imitator_pred_logits = self.imitator_pred_head(state_h)
        imitator_prey_logits = self.imitator_prey_head(state_h)
        critic_values = self.critic_head(state_h)
        return actor_logits, imitator_pred_logits, imitator_prey_logits, critic_values

    @torch.no_grad()
    def get_action(
        self,
        epi_states,
        epi_pred_opp_acts,
        epi_prey_opp_acts,
        step_states,
        step_self_acts,
        step_pred_opp_acts,
        step_prey_opp_acts,
        step_rtgs,
        step_timesteps,
        temperature: float = 1.0,
    ):
        self.eval()
        actor_logits, _, _, _ = self.forward(
            epi_states,
            epi_pred_opp_acts,
            epi_prey_opp_acts,
            step_states,
            step_self_acts,
            step_pred_opp_acts,
            step_prey_opp_acts,
            step_rtgs,
            step_timesteps,
        )
        query_logits = actor_logits[:, -1, :]
        probs = F.softmax(query_logits / temperature, dim=-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return action, query_logits

    @torch.no_grad()
    def get_opp_action(
        self,
        epi_states,
        epi_pred_opp_acts,
        epi_prey_opp_acts,
        step_states,
        step_self_acts,
        step_pred_opp_acts,
        step_prey_opp_acts,
        step_rtgs,
        step_timesteps,
        temperature: float = 1.0,
    ):
        self.eval()
        _, pred_logits, prey_logits, _ = self.forward(
            epi_states,
            epi_pred_opp_acts,
            epi_prey_opp_acts,
            step_states,
            step_self_acts,
            step_pred_opp_acts,
            step_prey_opp_acts,
            step_rtgs,
            step_timesteps,
        )
        pred_query = pred_logits[:, -1, :]
        prey_query = prey_logits[:, -1, :]

        pred_probs = F.softmax(pred_query / temperature, dim=-1)
        prey_probs = F.softmax(prey_query / temperature, dim=-1)

        pred_action = torch.multinomial(pred_probs, num_samples=1).squeeze(-1)
        prey_action = torch.multinomial(prey_probs, num_samples=1).squeeze(-1)
        return pred_action, prey_action, pred_query, prey_query

    @torch.no_grad()
    def get_value(
        self,
        epi_states,
        epi_pred_opp_acts,
        epi_prey_opp_acts,
        step_states,
        step_self_acts,
        step_pred_opp_acts,
        step_prey_opp_acts,
        step_rtgs,
        step_timesteps,
    ):
        self.eval()
        _, _, _, critic_values = self.forward(
            epi_states,
            epi_pred_opp_acts,
            epi_prey_opp_acts,
            step_states,
            step_self_acts,
            step_pred_opp_acts,
            step_prey_opp_acts,
            step_rtgs,
            step_timesteps,
        )
        return critic_values[:, -1, 0]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Maintains D_epi and D_step context for one predator perspective."""

    def __init__(self, h_epi: int = H_EPI, c_epi: int = C_EPI, b_seq: int = B_SEQ, device: str = "cpu"):
        self.h_epi = h_epi
        self.c_epi = c_epi
        self.b_seq = b_seq
        self.device = device

        self.epi_states = []
        self.epi_pred_opp_acts = []
        self.epi_prey_opp_acts = []

        self.step_states = []
        self.step_self_acts = []
        self.step_pred_opp_acts = []
        self.step_prey_opp_acts = []
        self.step_rtgs = []
        self.step_timesteps = []

    def set_epi_context(self, historical_trajectories):
        """
        Build D_epi from historical trajectories.

        Each trajectory entry must be:
          (state, pred_opp_act, prey_opp_act)
        """
        seg_len = self.h_epi // self.c_epi
        epi_states = []
        epi_pred_acts = []
        epi_prey_acts = []

        for traj in historical_trajectories[: self.c_epi]:
            if not traj:
                continue
            if len(traj) < seg_len:
                traj = traj + [traj[-1]] * (seg_len - len(traj))
            max_start = len(traj) - seg_len
            start = torch.randint(0, max(1, max_start + 1), (1,)).item()
            segment = traj[start : start + seg_len]
            for state, pred_act, prey_act in segment:
                epi_states.append(state)
                epi_pred_acts.append(int(pred_act))
                epi_prey_acts.append(int(prey_act))

        while len(epi_states) < self.h_epi:
            epi_states.append(epi_states[-1] if epi_states else torch.zeros(STATE_DIM))
            epi_pred_acts.append(epi_pred_acts[-1] if epi_pred_acts else 0)
            epi_prey_acts.append(epi_prey_acts[-1] if epi_prey_acts else 0)

        self.epi_states = epi_states[: self.h_epi]
        self.epi_pred_opp_acts = epi_pred_acts[: self.h_epi]
        self.epi_prey_opp_acts = epi_prey_acts[: self.h_epi]

    def reset_step_context(self):
        self.step_states = []
        self.step_self_acts = []
        self.step_pred_opp_acts = []
        self.step_prey_opp_acts = []
        self.step_rtgs = []
        self.step_timesteps = []

    def add_step(self, state, self_act: int, pred_opp: int, prey_opp: int, rtg: float, t: int):
        self.step_states.append(state)
        self.step_self_acts.append(int(self_act))
        self.step_pred_opp_acts.append(int(pred_opp))
        self.step_prey_opp_acts.append(int(prey_opp))
        self.step_rtgs.append(float(rtg))
        self.step_timesteps.append(int(t))

    def get_input_tensors(self, query_state, query_timestep: int):
        import torch

        def to_tensor(value):
            if isinstance(value, torch.Tensor):
                return value.float()
            return torch.tensor(value, dtype=torch.float32)

        device = self.device

        if not self.epi_states:
            epi_s = torch.zeros(1, self.h_epi, STATE_DIM, device=device)
            epi_pred = torch.zeros(1, self.h_epi, dtype=torch.long, device=device)
            epi_prey = torch.zeros(1, self.h_epi, dtype=torch.long, device=device)
        else:
            epi_s = torch.stack([to_tensor(s) for s in self.epi_states], dim=0).unsqueeze(0).to(device)
            epi_pred = torch.tensor(self.epi_pred_opp_acts, dtype=torch.long, device=device).unsqueeze(0)
            epi_prey = torch.tensor(self.epi_prey_opp_acts, dtype=torch.long, device=device).unsqueeze(0)

        n_steps = len(self.step_states)
        start = max(0, n_steps - (self.b_seq - 1))

        step_s_list = self.step_states[start:] + [query_state]
        step_ts_list = self.step_timesteps[start:] + [int(query_timestep)]

        step_states_t = torch.stack([to_tensor(s) for s in step_s_list], dim=0).unsqueeze(0).to(device)
        step_timesteps_t = torch.tensor(step_ts_list, dtype=torch.long, device=device).unsqueeze(0)

        step_sa_list = self.step_self_acts[start:]
        step_pred_list = self.step_pred_opp_acts[start:]
        step_prey_list = self.step_prey_opp_acts[start:]
        step_rtg_list = self.step_rtgs[start:]

        if not step_sa_list:
            step_sa_t = torch.zeros(1, 0, dtype=torch.long, device=device)
            step_pred_t = torch.zeros(1, 0, dtype=torch.long, device=device)
            step_prey_t = torch.zeros(1, 0, dtype=torch.long, device=device)
            step_rtg_t = torch.zeros(1, 0, 1, dtype=torch.float32, device=device)
        else:
            step_sa_t = torch.tensor(step_sa_list, dtype=torch.long, device=device).unsqueeze(0)
            step_pred_t = torch.tensor(step_pred_list, dtype=torch.long, device=device).unsqueeze(0)
            step_prey_t = torch.tensor(step_prey_list, dtype=torch.long, device=device).unsqueeze(0)
            step_rtg_t = torch.tensor(step_rtg_list, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)

        return {
            "epi_states": epi_s,
            "epi_pred_opp_acts": epi_pred,
            "epi_prey_opp_acts": epi_prey,
            "step_states": step_states_t,
            "step_self_acts": step_sa_t,
            "step_pred_opp_acts": step_pred_t,
            "step_prey_opp_acts": step_prey_t,
            "step_rtgs": step_rtg_t,
            "step_timesteps": step_timesteps_t,
        }


# ---------------------------------------------------------------------------
# Decision-time search
# ---------------------------------------------------------------------------


class DecisionTimeSearch:
    """DTS with split opponent modeling heads."""

    def __init__(
        self,
        model,
        env_model,
        n_self_actions: int = N_SELF_ACTIONS,
        M: int = 2,
        L: int = 1,
        gamma_search: float = 0.7,
        epsilon: float = 0.0,
        device: str = "cpu",
    ):
        self.model = model
        self.env_model = env_model
        self.n_self_actions = n_self_actions
        self.M = M
        self.L = L
        self.gamma_search = gamma_search
        self.epsilon = epsilon
        self.device = device

    @torch.no_grad()
    def select_action(self, context_builder: ContextBuilder, query_state, query_timestep: int):
        self.model.eval()
        
        # Get actor policy to filter top actions
        ctx_main = context_builder.get_input_tensors(query_state, query_timestep)
        _, actor_logits = self.model.get_action(**ctx_main)
        probs = torch.softmax(actor_logits, dim=-1).squeeze(0)
        
        # Only search Top-3 actions to save compute
        top_k_indices = torch.topk(probs, k=min(3, self.n_self_actions)).indices.cpu().numpy()
        
        q_values = {}
        for candidate in top_k_indices:
            candidate = int(candidate)
            q_sum = 0.0
            for _ in range(self.M):
                cumulative_r = 0.0
                gamma_acc = 1.0
                current_state = query_state
                current_t = int(query_timestep)

                sim_step_states = list(context_builder.step_states)
                sim_step_self_acts = list(context_builder.step_self_acts)
                sim_step_pred_opp_acts = list(context_builder.step_pred_opp_acts)
                sim_step_prey_opp_acts = list(context_builder.step_prey_opp_acts)
                sim_step_rtgs = list(context_builder.step_rtgs)
                sim_step_timesteps = list(context_builder.step_timesteps)

                for rollout_step in range(self.L):
                    sim_ctx = self._build_sim_context(
                        context_builder.epi_states,
                        context_builder.epi_pred_opp_acts,
                        context_builder.epi_prey_opp_acts,
                        sim_step_states,
                        sim_step_self_acts,
                        sim_step_pred_opp_acts,
                        sim_step_prey_opp_acts,
                        sim_step_rtgs,
                        sim_step_timesteps,
                        current_state,
                        current_t,
                    )

                    if rollout_step == 0:
                        sim_self_act = candidate
                    else:
                        act, _ = self.model.get_action(**sim_ctx)
                        sim_self_act = int(act.item())

                    pred_opp_act, prey_opp_act, _, _ = self.model.get_opp_action(**sim_ctx)
                    pred_opp_idx = int(pred_opp_act.item())
                    prey_opp_idx = int(prey_opp_act.item())

                    next_state, reward = self.env_model(
                        current_state,
                        sim_self_act,
                        pred_opp_idx,
                        prey_opp_idx,
                    )

                    cumulative_r += gamma_acc * reward
                    gamma_acc *= self.gamma_search

                    sim_step_states.append(current_state)
                    sim_step_self_acts.append(sim_self_act)
                    sim_step_pred_opp_acts.append(pred_opp_idx)
                    sim_step_prey_opp_acts.append(prey_opp_idx)
                    sim_step_rtgs.append(0.0)
                    sim_step_timesteps.append(current_t)

                    current_state = next_state
                    current_t += 1

                final_ctx = self._build_sim_context(
                    context_builder.epi_states,
                    context_builder.epi_pred_opp_acts,
                    context_builder.epi_prey_opp_acts,
                    sim_step_states,
                    sim_step_self_acts,
                    sim_step_pred_opp_acts,
                    sim_step_prey_opp_acts,
                    sim_step_rtgs,
                    sim_step_timesteps,
                    current_state,
                    current_t,
                )
                value_final = float(self.model.get_value(**final_ctx).item())
                q_rollout = cumulative_r + (self.gamma_search ** (self.L + 1)) * value_final
                q_sum += q_rollout

            q_values[candidate] = q_sum / self.M

        best_action = max(q_values, key=q_values.get)
        best_q = q_values[best_action]

        if abs(best_q) > self.epsilon:
            return best_action, best_q, "search"

        ctx = context_builder.get_input_tensors(query_state, query_timestep)
        act, _ = self.model.get_action(**ctx)
        return int(act.item()), best_q, "actor"

    def _build_sim_context(
        self,
        epi_states,
        epi_pred_opp_acts,
        epi_prey_opp_acts,
        sim_step_states,
        sim_step_self_acts,
        sim_step_pred_opp_acts,
        sim_step_prey_opp_acts,
        sim_step_rtgs,
        sim_step_timesteps,
        current_state,
        current_t,
    ):
        import torch

        def to_tensor(value):
            if isinstance(value, torch.Tensor):
                return value.float()
            return torch.tensor(value, dtype=torch.float32)

        device = self.device

        if not epi_states:
            epi_s = torch.zeros(1, H_EPI, STATE_DIM, device=device)
            epi_pred = torch.zeros(1, H_EPI, dtype=torch.long, device=device)
            epi_prey = torch.zeros(1, H_EPI, dtype=torch.long, device=device)
        else:
            epi_s = torch.stack([to_tensor(s) for s in epi_states], dim=0).unsqueeze(0).to(device)
            epi_pred = torch.tensor(epi_pred_opp_acts, dtype=torch.long, device=device).unsqueeze(0)
            epi_prey = torch.tensor(epi_prey_opp_acts, dtype=torch.long, device=device).unsqueeze(0)

        n_steps = len(sim_step_states)
        start = max(0, n_steps - (B_SEQ - 1))

        ss_list = sim_step_states[start:] + [current_state]
        st_list = sim_step_timesteps[start:] + [current_t]

        step_states_t = torch.stack([to_tensor(s) for s in ss_list], dim=0).unsqueeze(0).to(device)
        step_ts_t = torch.tensor(st_list, dtype=torch.long, device=device).unsqueeze(0)

        sa_list = sim_step_self_acts[start:]
        pred_list = sim_step_pred_opp_acts[start:]
        prey_list = sim_step_prey_opp_acts[start:]
        rtg_list = sim_step_rtgs[start:]

        if not sa_list:
            step_sa_t = torch.zeros(1, 0, dtype=torch.long, device=device)
            step_pred_t = torch.zeros(1, 0, dtype=torch.long, device=device)
            step_prey_t = torch.zeros(1, 0, dtype=torch.long, device=device)
            step_rtg_t = torch.zeros(1, 0, 1, dtype=torch.float32, device=device)
        else:
            step_sa_t = torch.tensor(sa_list, dtype=torch.long, device=device).unsqueeze(0)
            step_pred_t = torch.tensor(pred_list, dtype=torch.long, device=device).unsqueeze(0)
            step_prey_t = torch.tensor(prey_list, dtype=torch.long, device=device).unsqueeze(0)
            step_rtg_t = torch.tensor(rtg_list, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)

        return {
            "epi_states": epi_s,
            "epi_pred_opp_acts": epi_pred,
            "epi_prey_opp_acts": epi_prey,
            "step_states": step_states_t,
            "step_self_acts": step_sa_t,
            "step_pred_opp_acts": step_pred_t,
            "step_prey_opp_acts": step_prey_t,
            "step_rtgs": step_rtg_t,
            "step_timesteps": step_ts_t,
        }


if __name__ == "__main__":
    batch_size, h_len, l_len = 2, H_EPI, 5
    model = OMISModel()
    print(f"OMISModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    epi_states = torch.randn(batch_size, h_len, STATE_DIM)
    epi_pred = torch.randint(0, N_PRED_OPP_ACTIONS, (batch_size, h_len))
    epi_prey = torch.randint(0, N_PREY_ACTIONS, (batch_size, h_len))

    step_states = torch.randn(batch_size, l_len, STATE_DIM)
    step_self = torch.randint(0, N_SELF_ACTIONS, (batch_size, l_len - 1))
    step_pred = torch.randint(0, N_PRED_OPP_ACTIONS, (batch_size, l_len - 1))
    step_prey = torch.randint(0, N_PREY_ACTIONS, (batch_size, l_len - 1))
    step_rtgs = torch.randn(batch_size, l_len - 1, 1)
    step_ts = torch.arange(l_len).unsqueeze(0).expand(batch_size, -1)

    actor_logits, pred_logits, prey_logits, critic_values = model(
        epi_states,
        epi_pred,
        epi_prey,
        step_states,
        step_self,
        step_pred,
        step_prey,
        step_rtgs,
        step_ts,
    )

    print(f"actor logits shape: {actor_logits.shape}")
    print(f"pred opp logits shape: {pred_logits.shape}")
    print(f"prey opp logits shape: {prey_logits.shape}")
    print(f"critic values shape: {critic_values.shape}")
