"""
omisAgent.py
============
OMIS Transformer model for Malmo predator-prey environment.
Implements the GPT2-style causal Transformer from Appendix F of:
  "Opponent Modeling with In-context Search" (Jing et al., NeurIPS 2024)

Environment mapping:
  - Self-agent  : Prey   (agent index 2)
  - Opponents   : Predator1 (index 0) + Predator2 (index 1)
  - State dim   : 128  (VoxelEncoder output)
  - Action dim  : 8    (flattened multidiscrete: 3 move * 3 turn * 2 attack -> 18,
                        but we use 8 = move(3) + turn(3) + attack(2) as separate heads
                        and encode as a single integer index 0..17 for the Transformer)
  - Opp actions : 2 opponents × 8 = 16 dims total, encoded as joint integer index 0..323

Architecture (Appendix F):
  - Backbone: 3-block GPT2 decoder, 1 attention head, hidden=32, FF=128 (GELU), dropout=0.1
  - Modality encoders: Linear(state_dim, 32) + LeakyReLU
                       Linear(act_dim, 32)   (no activation)
                       Linear(1, 32)         (RTG, no activation)
  - Output heads: Linear(32, n_self_actions) for actor
                  Linear(32, n_opp_actions)  for imitator
                  Linear(32, 1)              for critic
  - Positional timestep encoding (sinusoidal, same as Decision Transformer)
  - Agent index encoding added to each token

Sequence format (pretraining, timestep t):
  D_epi tokens: (s̃1,ã⁻¹1), ..., (s̃H,ã⁻¹H)          [H=15 pairs → 30 tokens]
  D_step tokens: st-B+1, a1t-B+1, a⁻¹t-B+1, Gt-B+1, ... st-1, a1t-1, a⁻¹t-1, Gt-1, st
                 [B=20 steps → up to 4*B tokens for the step window, ends with current state]
  Total max context: 2*H + 4*B = 30 + 80 = 110 tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Hyperparameters (from Appendix F and H.2)
# ─────────────────────────────────────────────────────────────────

STATE_DIM       = 128    # VoxelEncoder output dimension
N_SELF_ACTIONS  = 18     # 3 move * 3 turn * 2 attack (flattened)
N_OPP_ACTIONS   = 18     # Per opponent (same action space as self-agent)
N_OPPONENTS     = 2      # Predator1 + Predator2

# GPT2 backbone hyperparameters (Appendix F + H.2)
D_MODEL         = 32     # Hidden dim of all layers except FF
D_FF            = 128    # Feed-forward expansion dim (uses GELU)
N_HEADS         = 1      # Single attention head
N_BLOCKS        = 3      # Self-attention blocks
DROPOUT         = 0.1    # Applied to residual and attention weights

# Sequence construction hyperparameters (H.2)
H_EPI           = 15     # Sequence length of D_epi (state-action pairs)
C_EPI           = 3      # Number of trajectories sampled to build D_epi
B_SEQ           = 20     # Max step-wise window length for Transformer

# Token types for agent index encoding
TOKEN_STATE     = 0
TOKEN_SELF_ACT  = 1
TOKEN_OPP_ACT   = 2
TOKEN_RTG       = 3
TOKEN_EPI_STATE = 4
TOKEN_EPI_ACT   = 5
N_TOKEN_TYPES   = 6


# ─────────────────────────────────────────────────────────────────
# Sinusoidal timestep encoding (Chen et al. 2021 / Decision Transformer)
# ─────────────────────────────────────────────────────────────────

def get_timestep_encoding(timesteps, d_model):
    """
    Sinusoidal positional encoding for episodic timesteps.
    timesteps: (B, T) integer tensor of timestep indices
    returns  : (B, T, d_model) float tensor
    """
    B, T = timesteps.shape
    half = d_model // 2
    div  = torch.exp(
        torch.arange(half, dtype=torch.float32, device=timesteps.device)
        * -(math.log(10000.0) / half)
    )  # (half,)
    ts = timesteps.float().unsqueeze(-1)  # (B, T, 1)
    sin_enc = torch.sin(ts * div)         # (B, T, half)
    cos_enc = torch.cos(ts * div)         # (B, T, half)
    enc = torch.cat([sin_enc, cos_enc], dim=-1)  # (B, T, d_model)
    return enc


# ─────────────────────────────────────────────────────────────────
# GPT2-style Transformer block
# ─────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention with dropout."""

    def __init__(self, d_model, dropout, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.res_drop  = nn.Dropout(dropout)
        # causal mask — upper triangular
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        """x: (B, T, d_model)"""
        B, T, C = x.shape
        Q = self.q_proj(x)  # (B, T, C)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scale  = math.sqrt(C)
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale  # (B, T, T)
        scores = scores.masked_fill(self.causal_mask[:T, :T].unsqueeze(0), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        out = torch.bmm(weights, V)                # (B, T, C)
        out = self.res_drop(self.out_proj(out))
        return out


class TransformerBlock(nn.Module):
    """GPT2-style block: LayerNorm → Attention + residual → LayerNorm → FF + residual."""

    def __init__(self, d_model, d_ff, dropout, max_len=512):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, dropout, max_len)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────
# OMIS Model
# ─────────────────────────────────────────────────────────────────

class OMISModel(nn.Module):
    """
    OMIS causal Transformer with three in-context components:
      πθ   (actor)    — predicts self-agent action given (state, D_t)
      µϕ   (imitator) — predicts opponent joint action given (state, D_t)
      Vω   (critic)   — predicts RTG value given (state, D_t)

    Input construction:
      The caller builds two token streams and passes them together:
        1. epi_states  : (B, H, state_dim)   episode-wise in-context states
        2. epi_opp_acts: (B, H, N_OPP_ACTIONS*N_OPPONENTS) episode-wise opp actions
                         (encoded as joint one-hot or embedding index)
        3. step_states : (B, L, state_dim)   step-wise states (current episode)
        4. step_self_acts: (B, L, 1)         step-wise self-agent actions (int)
        5. step_opp_acts : (B, L, 1)         step-wise opponent joint actions (int)
        6. step_rtgs   : (B, L, 1)           step-wise RTGs
        7. step_timesteps: (B, L)            integer timestep indices
        where L <= B_SEQ and the final state token is the query state s_t.

    Token ordering in the sequence (following Appendix F):
      [s̃1, ã⁻¹1, ..., s̃H, ã⁻¹H,     ← D_epi: 2H tokens
       s_{t-B+1}, a1_{t-B+1}, a⁻¹_{t-B+1}, G_{t-B+1},   ← 4 tokens per step
       ...,
       s_{t-1}, a1_{t-1}, a⁻¹_{t-1}, G_{t-1},
       s_t]                             ← query state, 1 token

    Output positions: the model predicts at every state token position.
      At s_t: actor predicts a1_t, imitator predicts a⁻¹_t, critic predicts V_t
    """

    def __init__(
        self,
        state_dim       = STATE_DIM,
        n_self_actions  = N_SELF_ACTIONS,
        n_opp_actions   = N_OPP_ACTIONS,
        n_opponents     = N_OPPONENTS,
        d_model         = D_MODEL,
        d_ff            = D_FF,
        n_heads         = N_HEADS,
        n_blocks        = N_BLOCKS,
        dropout         = DROPOUT,
        h_epi           = H_EPI,
        b_seq           = B_SEQ,
        max_len         = 512,
    ):
        super().__init__()

        self.state_dim      = state_dim
        self.n_self_actions = n_self_actions
        self.n_opp_actions  = n_opp_actions
        self.n_opponents    = n_opponents
        self.d_model        = d_model
        self.h_epi          = h_epi
        self.b_seq          = b_seq

        # Joint opponent action space size
        self.n_joint_opp = n_opp_actions ** n_opponents  # 18^2 = 324

        # ── Modality-specific encoders (Appendix F) ──────────────────
        # States: Linear + LeakyReLU → d_model
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LeakyReLU(),
        )
        # Self-agent actions: embedding → d_model (no activation)
        self.self_act_encoder = nn.Embedding(n_self_actions, d_model)

        # Opponent joint actions: embedding → d_model (no activation)
        self.opp_act_encoder = nn.Embedding(self.n_joint_opp, d_model)

        # RTG: Linear(1, d_model) (no activation)
        self.rtg_encoder = nn.Linear(1, d_model)

        # Agent index encoding (TOKEN_TYPES → d_model)
        self.token_type_emb = nn.Embedding(N_TOKEN_TYPES, d_model)

        # ── GPT2 backbone ─────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, dropout, max_len)
            for _ in range(n_blocks)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        # ── Output heads (Appendix F: all linear) ────────────────────
        # Actor πθ: predicts self-agent action
        self.actor_head = nn.Linear(d_model, n_self_actions)
        # Imitator µϕ: predicts joint opponent action
        self.imitator_head = nn.Linear(d_model, self.n_joint_opp)
        # Critic Vω: predicts scalar value
        self.critic_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Encode a full sequence and return the Transformer output ─────

    def _build_sequence(
        self,
        epi_states,       # (B, H, state_dim)
        epi_opp_acts,     # (B, H,) long  — joint opponent action indices
        step_states,      # (B, L, state_dim)
        step_self_acts,   # (B, L-1,) long   (no action for the final query state)
        step_opp_acts,    # (B, L-1,) long
        step_rtgs,        # (B, L-1, 1) float
        step_timesteps,   # (B, L,) long
    ):
        """
        Build the full token sequence and return (token_embeddings, timestep_enc).

        D_epi part: for each h in [H]: state token, opp_act token  → 2H tokens
        D_step part: for each step i in [L-1]: state, self_act, opp_act, rtg → 4*(L-1) tokens
                     final query state: 1 token
        Total tokens = 2H + 4*(L-1) + 1

        Returns:
          tokens : (B, T_total, d_model)
          ttypes : (B, T_total) — token type indices for agent-index encoding
          tsteps : (B, T_total) — timestep index for each token
        """
        B    = epi_states.size(0)
        H    = epi_states.size(1)
        L    = step_states.size(1)   # L includes the query state at the end
        device = epi_states.device

        token_list  = []
        ttype_list  = []
        tstep_list  = []

        # ── D_epi tokens ─────────────────────────────────────────────
        for h in range(H):
            s_emb   = self.state_encoder(epi_states[:, h, :])           # (B, d_model)
            oa_emb  = self.opp_act_encoder(epi_opp_acts[:, h])           # (B, d_model)
            token_list.extend([s_emb, oa_emb])
            ttype_list.extend([
                torch.full((B,), TOKEN_EPI_STATE, dtype=torch.long, device=device),
                torch.full((B,), TOKEN_EPI_ACT,   dtype=torch.long, device=device),
            ])
            # D_epi tokens use timestep 0 (they are from past episodes)
            tstep_list.extend([
                torch.zeros(B, dtype=torch.long, device=device),
                torch.zeros(B, dtype=torch.long, device=device),
            ])

        # ── D_step tokens (all steps except the final query state) ───
        for i in range(L - 1):
            s_emb   = self.state_encoder(step_states[:, i, :])          # (B, d_model)
            sa_emb  = self.self_act_encoder(step_self_acts[:, i])        # (B, d_model)
            oa_emb  = self.opp_act_encoder(step_opp_acts[:, i])          # (B, d_model)
            rtg_emb = self.rtg_encoder(step_rtgs[:, i, :])               # (B, d_model)
            token_list.extend([s_emb, sa_emb, oa_emb, rtg_emb])
            ttype_list.extend([
                torch.full((B,), TOKEN_STATE,    dtype=torch.long, device=device),
                torch.full((B,), TOKEN_SELF_ACT, dtype=torch.long, device=device),
                torch.full((B,), TOKEN_OPP_ACT,  dtype=torch.long, device=device),
                torch.full((B,), TOKEN_RTG,      dtype=torch.long, device=device),
            ])
            t = step_timesteps[:, i]  # (B,)
            tstep_list.extend([t, t, t, t])

        # ── Final query state token ───────────────────────────────────
        s_emb  = self.state_encoder(step_states[:, L - 1, :])           # (B, d_model)
        token_list.append(s_emb)
        ttype_list.append(torch.full((B,), TOKEN_STATE, dtype=torch.long, device=device))
        tstep_list.append(step_timesteps[:, L - 1])

        # Stack: each element is (B, d_model) → (B, T_total, d_model)
        tokens  = torch.stack(token_list,  dim=1)   # (B, T_total, d_model)
        ttypes  = torch.stack(ttype_list,  dim=1)   # (B, T_total)
        tsteps  = torch.stack(tstep_list,  dim=1)   # (B, T_total)

        return tokens, ttypes, tsteps

    def _encode(self, tokens, ttypes, tsteps):
        """
        Add token-type and timestep encodings, then run through Transformer blocks.
        tokens : (B, T, d_model)
        ttypes : (B, T)
        tsteps : (B, T)
        returns: (B, T, d_model)
        """
        # Agent index (token type) encoding
        ttype_enc = self.token_type_emb(ttypes)         # (B, T, d_model)

        # Positional timestep encoding
        tstep_enc = get_timestep_encoding(tsteps, self.d_model)  # (B, T, d_model)

        x = tokens + ttype_enc + tstep_enc

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        return x

    def forward(
        self,
        epi_states,       # (B, H, state_dim)
        epi_opp_acts,     # (B, H,) long
        step_states,      # (B, L, state_dim)
        step_self_acts,   # (B, L-1,) long
        step_opp_acts,    # (B, L-1,) long
        step_rtgs,        # (B, L-1, 1) float
        step_timesteps,   # (B, L,) long
    ):
        """
        Forward pass — returns predictions at all state token positions.

        Returns:
          actor_logits   : (B, T_states, n_self_actions)
          imitator_logits: (B, T_states, n_joint_opp)
          critic_values  : (B, T_states, 1)

        where T_states is the number of state tokens = H + L (epi states + step states).
        The last entry [:, -1, :] corresponds to the query state s_t.
        """
        tokens, ttypes, tsteps = self._build_sequence(
            epi_states, epi_opp_acts,
            step_states, step_self_acts, step_opp_acts, step_rtgs, step_timesteps,
        )

        h = self._encode(tokens, ttypes, tsteps)  # (B, T_total, d_model)

        # Extract outputs at state token positions
        # D_epi has 2H tokens: state at positions 0, 2, 4, ..., 2H-2
        # D_step has 4*(L-1) + 1 tokens: state at positions 2H+0, 2H+4, ..., 2H+4*(L-1)
        H = epi_states.size(1)
        L = step_states.size(1)

        epi_state_positions  = list(range(0, 2 * H, 2))             # [0, 2, ..., 2H-2]
        step_state_positions = [2 * H + 4 * i for i in range(L)]    # step states
        state_positions      = epi_state_positions + step_state_positions

        state_h = h[:, state_positions, :]   # (B, H+L, d_model)

        actor_logits    = self.actor_head(state_h)     # (B, H+L, n_self_actions)
        imitator_logits = self.imitator_head(state_h)  # (B, H+L, n_joint_opp)
        critic_values   = self.critic_head(state_h)    # (B, H+L, 1)

        return actor_logits, imitator_logits, critic_values

    # ── Inference helpers ────────────────────────────────────────────

    @torch.no_grad()
    def get_action(
        self,
        epi_states, epi_opp_acts,
        step_states, step_self_acts, step_opp_acts, step_rtgs, step_timesteps,
        temperature=1.0,
    ):
        """
        Sample self-agent action from actor πθ at the query state (last position).
        Returns: action_idx (int), action_logits (tensor)
        """
        self.eval()
        actor_logits, _, _ = self.forward(
            epi_states, epi_opp_acts,
            step_states, step_self_acts, step_opp_acts, step_rtgs, step_timesteps,
        )
        query_logits = actor_logits[:, -1, :]   # (B, n_self_actions)
        probs        = F.softmax(query_logits / temperature, dim=-1)
        action       = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return action, query_logits

    @torch.no_grad()
    def get_opp_action(
        self,
        epi_states, epi_opp_acts,
        step_states, step_self_acts, step_opp_acts, step_rtgs, step_timesteps,
        temperature=1.0,
    ):
        """
        Sample opponent joint action from imitator µϕ at the query state.
        Returns: joint_action_idx (int), imitator_logits (tensor)
        """
        self.eval()
        _, imitator_logits, _ = self.forward(
            epi_states, epi_opp_acts,
            step_states, step_self_acts, step_opp_acts, step_rtgs, step_timesteps,
        )
        query_logits = imitator_logits[:, -1, :]  # (B, n_joint_opp)
        probs        = F.softmax(query_logits / temperature, dim=-1)
        joint_action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return joint_action, query_logits

    @torch.no_grad()
    def get_value(
        self,
        epi_states, epi_opp_acts,
        step_states, step_self_acts, step_opp_acts, step_rtgs, step_timesteps,
    ):
        """
        Estimate state value from critic Vω at the query state.
        Returns: value (float tensor, shape (B,))
        """
        self.eval()
        _, _, critic_values = self.forward(
            epi_states, epi_opp_acts,
            step_states, step_self_acts, step_opp_acts, step_rtgs, step_timesteps,
        )
        return critic_values[:, -1, 0]   # (B,)


# ─────────────────────────────────────────────────────────────────
# Action encoding/decoding utilities
# ─────────────────────────────────────────────────────────────────

def encode_self_action(move, turn, attack):
    """
    Convert multidiscrete action (move, turn, attack) to flat index.
    move   : 0=forward, 1=backward, 2=stop
    turn   : 0=left, 1=right, 2=none
    attack : 0=yes, 1=no
    Returns int in [0, 17]
    """
    return move * 6 + turn * 2 + attack


def decode_self_action(idx):
    """Convert flat action index back to (move, turn, attack)."""
    attack = idx % 2
    turn   = (idx // 2) % 3
    move   = idx // 6
    return move, turn, attack


def encode_joint_opp_action(act0, act1):
    """
    Encode two opponent actions (each in [0,17]) into a joint index [0, 323].
    act0: action of Predator1 (flat index 0-17)
    act1: action of Predator2 (flat index 0-17)
    """
    return act0 * N_OPP_ACTIONS + act1


def decode_joint_opp_action(joint_idx):
    """Decode joint opponent index back to (act0, act1)."""
    act1 = joint_idx % N_OPP_ACTIONS
    act0 = joint_idx // N_OPP_ACTIONS
    return act0, act1


def action_list_to_malmo(flat_idx):
    """
    Convert flat action index (0-17) to Malmo command tuple
    (move_cmd, turn_cmd, attack_cmd).
    Returns indices into the MOVE_CMDS, TURN_CMDS, ATTACK_CMDS lists
    defined in malmoEnv.py.
    """
    move, turn, attack = decode_self_action(flat_idx)
    return move, turn, attack


# ─────────────────────────────────────────────────────────────────
# Context builder — manages D_epi and D_step buffers
# ─────────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Manages the in-context data (D_epi and D_step) for a single agent
    during an episode. Builds the input tensors for OMISModel.

    D_epi (episode-wise):  sampled once per episode from historical trajectories.
    D_step (step-wise):    grows during the current episode.
    """

    def __init__(self, h_epi=H_EPI, c_epi=C_EPI, b_seq=B_SEQ, device="cpu"):
        self.h_epi  = h_epi
        self.c_epi  = c_epi
        self.b_seq  = b_seq
        self.device = device

        # D_epi: stored as lists of (state, opp_joint_act) of length H
        self.epi_states   = []   # list of np/tensor shape (state_dim,)
        self.epi_opp_acts = []   # list of int (joint opp action index)

        # D_step: growing history of current episode
        self.step_states    = []   # list of np/tensor (state_dim,)
        self.step_self_acts = []   # list of int
        self.step_opp_acts  = []   # list of int
        self.step_rtgs      = []   # list of float
        self.step_timesteps = []   # list of int

    def set_epi_context(self, historical_trajectories):
        """
        Build D_epi from C historical trajectories (Appendix C).
        Each trajectory is a list of (state, opp_joint_act) tuples.
        Samples C consecutive segments of length H//C from each trajectory
        and concatenates them.

        historical_trajectories: list of C trajectories, each a list of
                                 (state_tensor, opp_joint_act_int) tuples.
        """
        seg_len = self.h_epi // self.c_epi  # H/C steps per trajectory segment
        epi_states   = []
        epi_opp_acts = []

        for traj in historical_trajectories[:self.c_epi]:
            if len(traj) < seg_len:
                # Pad with the last entry if trajectory is too short
                traj = traj + [traj[-1]] * (seg_len - len(traj))
            # Sample a random consecutive segment
            max_start = len(traj) - seg_len
            start = torch.randint(0, max(1, max_start + 1), (1,)).item()
            segment = traj[start: start + seg_len]
            for (s, a) in segment:
                epi_states.append(s)
                epi_opp_acts.append(a)

        # Truncate or pad to exactly H_EPI entries
        while len(epi_states) < self.h_epi:
            epi_states.append(epi_states[-1] if epi_states else torch.zeros(STATE_DIM))
            epi_opp_acts.append(epi_opp_acts[-1] if epi_opp_acts else 0)

        self.epi_states   = epi_states[:self.h_epi]
        self.epi_opp_acts = epi_opp_acts[:self.h_epi]

    def reset_step_context(self):
        """Call at the start of each episode."""
        self.step_states    = []
        self.step_self_acts = []
        self.step_opp_acts  = []
        self.step_rtgs      = []
        self.step_timesteps = []

    def add_step(self, state, self_act, opp_act, rtg, timestep):
        """
        Record a completed step (s, a1, a⁻¹, G, t).
        state   : tensor or array (state_dim,)
        self_act: int flat action index
        opp_act : int joint opponent action index
        rtg     : float return-to-go
        timestep: int
        """
        self.step_states.append(state)
        self.step_self_acts.append(self_act)
        self.step_opp_acts.append(opp_act)
        self.step_rtgs.append(rtg)
        self.step_timesteps.append(timestep)

    def get_input_tensors(self, query_state, query_timestep):
        """
        Build and return batched input tensors for OMISModel.forward().
        query_state    : tensor (state_dim,) — current state s_t
        query_timestep : int

        Returns dict with keys matching OMISModel.forward() signature,
        all with batch dimension B=1.
        """
        import torch
        import numpy as np

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.float()
            return torch.tensor(x, dtype=torch.float32)

        device = self.device

        # ── D_epi tensors ─────────────────────────────────────────────
        if len(self.epi_states) == 0:
            # No epi context yet — use zeros
            epi_s = torch.zeros(1, self.h_epi, STATE_DIM, device=device)
            epi_a = torch.zeros(1, self.h_epi, dtype=torch.long, device=device)
        else:
            epi_s = torch.stack([to_tensor(s) for s in self.epi_states], dim=0)
            epi_s = epi_s.unsqueeze(0).to(device)   # (1, H, state_dim)
            epi_a = torch.tensor(self.epi_opp_acts, dtype=torch.long, device=device)
            epi_a = epi_a.unsqueeze(0)               # (1, H)

        # ── D_step tensors — use last B_SEQ steps + query state ──────
        # Truncate to the last b_seq-1 steps (we append the query state)
        n_steps  = len(self.step_states)
        start    = max(0, n_steps - (self.b_seq - 1))

        step_s_list   = self.step_states[start:]
        step_sa_list  = self.step_self_acts[start:]
        step_oa_list  = self.step_opp_acts[start:]
        step_rtg_list = self.step_rtgs[start:]
        step_ts_list  = self.step_timesteps[start:]

        # Append the query state
        step_s_list  = step_s_list  + [query_state]
        step_ts_list = step_ts_list + [query_timestep]

        L = len(step_s_list)

        step_states_t = torch.stack([to_tensor(s) for s in step_s_list], dim=0)
        step_states_t = step_states_t.unsqueeze(0).to(device)     # (1, L, state_dim)

        step_timesteps_t = torch.tensor(step_ts_list, dtype=torch.long, device=device)
        step_timesteps_t = step_timesteps_t.unsqueeze(0)           # (1, L)

        if len(step_sa_list) == 0:
            # No step history yet — model only has the query state
            step_sa_t  = torch.zeros(1, 0, dtype=torch.long, device=device)
            step_oa_t  = torch.zeros(1, 0, dtype=torch.long, device=device)
            step_rtg_t = torch.zeros(1, 0, 1, dtype=torch.float32, device=device)
        else:
            step_sa_t = torch.tensor(step_sa_list, dtype=torch.long, device=device)
            step_sa_t = step_sa_t.unsqueeze(0)      # (1, L-1)

            step_oa_t = torch.tensor(step_oa_list, dtype=torch.long, device=device)
            step_oa_t = step_oa_t.unsqueeze(0)      # (1, L-1)

            step_rtg_t = torch.tensor(step_rtg_list, dtype=torch.float32, device=device)
            step_rtg_t = step_rtg_t.unsqueeze(0).unsqueeze(-1)  # (1, L-1, 1)

        return {
            "epi_states":      epi_s,
            "epi_opp_acts":    epi_a,
            "step_states":     step_states_t,
            "step_self_acts":  step_sa_t,
            "step_opp_acts":   step_oa_t,
            "step_rtgs":       step_rtg_t,
            "step_timesteps":  step_timesteps_t,
        }


# ─────────────────────────────────────────────────────────────────
# Decision-Time Search (Section 4.2, Algorithm 1 lines 16-21)
# ─────────────────────────────────────────────────────────────────

class DecisionTimeSearch:
    """
    Implements OMIS Decision-Time Search using the learned in-context components.

    At each real timestep t:
      For each legal self-agent action â1_t:
        Perform M rollouts of length L using πθ (actor) and µϕ (imitator)
        Estimate Q̂(s_t, â1_t) using Vω (critic) at the final search state
      Select action with max Q̂ as πsearch
      Mix with πθ using threshold ε (Eq. 10)

    DTS hyperparameters (H.3, PP environment values):
      M = 3 rollouts per action
      L = 3 steps per rollout
      γ_search = 0.7
      ε = 10   (mixing threshold)
    """

    def __init__(
        self,
        model,            # OMISModel instance (pretrained)
        env_model,        # callable: (state, self_act, opp_act) → (next_state, reward)
        n_self_actions = N_SELF_ACTIONS,
        M              = 3,       # rollouts per action
        L              = 3,       # rollout length
        gamma_search   = 0.7,
        epsilon        = 10.0,    # mixing threshold
        device         = "cpu",
    ):
        self.model          = model
        self.env_model      = env_model
        self.n_self_actions = n_self_actions
        self.M              = M
        self.L              = L
        self.gamma_search   = gamma_search
        self.epsilon        = epsilon
        self.device         = device

    @torch.no_grad()
    def select_action(self, context_builder, query_state, query_timestep):
        """
        Run DTS and return the selected action index (int).

        context_builder: ContextBuilder with current D_epi and D_step filled
        query_state    : np array or tensor (state_dim,)
        query_timestep : int
        """
        self.model.eval()
        q_values = {}

        for a_candidate in range(self.n_self_actions):
            q_sum = 0.0

            for _ in range(self.M):
                # Get current context tensors
                ctx = context_builder.get_input_tensors(query_state, query_timestep)

                cumulative_r = 0.0
                gamma_acc    = 1.0
                current_state = query_state
                current_t     = query_timestep

                # Make a local copy of the step context for simulation
                sim_step_states    = list(context_builder.step_states)
                sim_step_self_acts = list(context_builder.step_self_acts)
                sim_step_opp_acts  = list(context_builder.step_opp_acts)
                sim_step_rtgs      = list(context_builder.step_rtgs)
                sim_step_timesteps = list(context_builder.step_timesteps)

                for l in range(self.L):
                    # Build context for this rollout step
                    sim_ctx = self._build_sim_context(
                        context_builder.epi_states,
                        context_builder.epi_opp_acts,
                        sim_step_states,
                        sim_step_self_acts,
                        sim_step_opp_acts,
                        sim_step_rtgs,
                        sim_step_timesteps,
                        current_state,
                        current_t,
                    )

                    # Choose action for this rollout step
                    if l == 0:
                        # First step: use the candidate action
                        sim_self_act = a_candidate
                    else:
                        # Subsequent steps: use actor πθ
                        act, _ = self.model.get_action(**sim_ctx)
                        sim_self_act = act.item()

                    # Use imitator µϕ to sample opponent action
                    opp_act, _ = self.model.get_opp_action(**sim_ctx)
                    sim_opp_act = opp_act.item()

                    # Transition using environment model P
                    next_state, reward = self.env_model(
                        current_state, sim_self_act, sim_opp_act
                    )

                    cumulative_r += gamma_acc * reward
                    gamma_acc    *= self.gamma_search

                    # Update simulated context
                    sim_step_states.append(current_state)
                    sim_step_self_acts.append(sim_self_act)
                    sim_step_opp_acts.append(sim_opp_act)
                    sim_step_rtgs.append(0.0)   # placeholder RTG during search
                    sim_step_timesteps.append(current_t)

                    current_state = next_state
                    current_t    += 1

                # Estimate value of the final search state using Vω
                final_ctx = self._build_sim_context(
                    context_builder.epi_states,
                    context_builder.epi_opp_acts,
                    sim_step_states,
                    sim_step_self_acts,
                    sim_step_opp_acts,
                    sim_step_rtgs,
                    sim_step_timesteps,
                    current_state,
                    current_t,
                )
                v_final = self.model.get_value(**final_ctx)  # (1,)
                v_final = v_final.item()

                # Q̂ estimate for this rollout (Eq. 8)
                q_rollout = cumulative_r + (self.gamma_search ** (self.L + 1)) * v_final
                q_sum    += q_rollout

            q_values[a_candidate] = q_sum / self.M

        # πsearch: action with maximum Q̂ (Eq. 9)
        best_action = max(q_values, key=q_values.__getitem__)
        best_q      = q_values[best_action]

        # Mixing technique (Eq. 10): use πsearch if ||Q̂|| > ε, else use πθ
        if abs(best_q) > self.epsilon:
            return best_action, best_q, "search"
        else:
            ctx = context_builder.get_input_tensors(query_state, query_timestep)
            act, _ = self.model.get_action(**ctx)
            return act.item(), best_q, "actor"

    def _build_sim_context(
        self,
        epi_states, epi_opp_acts,
        sim_step_states, sim_step_self_acts, sim_step_opp_acts,
        sim_step_rtgs, sim_step_timesteps,
        current_state, current_t,
    ):
        """Build input tensors from simulated context for inference."""
        import numpy as np
        device = self.device

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.float()
            return torch.tensor(x, dtype=torch.float32)

        # D_epi
        if len(epi_states) == 0:
            epi_s = torch.zeros(1, H_EPI, STATE_DIM, device=device)
            epi_a = torch.zeros(1, H_EPI, dtype=torch.long, device=device)
        else:
            epi_s = torch.stack([to_tensor(s) for s in epi_states], dim=0)
            epi_s = epi_s.unsqueeze(0).to(device)
            epi_a = torch.tensor(epi_opp_acts, dtype=torch.long, device=device).unsqueeze(0)

        # D_step: use last b_seq-1 steps + current state
        n  = len(sim_step_states)
        st = max(0, n - (B_SEQ - 1))

        ss_list  = sim_step_states[st:]    + [current_state]
        sst_list = sim_step_timesteps[st:] + [current_t]
        L_sim    = len(ss_list)

        step_states_t = torch.stack([to_tensor(s) for s in ss_list]).unsqueeze(0).to(device)
        step_ts_t     = torch.tensor(sst_list, dtype=torch.long, device=device).unsqueeze(0)

        sa_list  = sim_step_self_acts[st:]
        oa_list  = sim_step_opp_acts[st:]
        rtg_list = sim_step_rtgs[st:]

        if len(sa_list) == 0:
            step_sa_t  = torch.zeros(1, 0, dtype=torch.long, device=device)
            step_oa_t  = torch.zeros(1, 0, dtype=torch.long, device=device)
            step_rtg_t = torch.zeros(1, 0, 1, device=device)
        else:
            step_sa_t  = torch.tensor(sa_list, dtype=torch.long, device=device).unsqueeze(0)
            step_oa_t  = torch.tensor(oa_list, dtype=torch.long, device=device).unsqueeze(0)
            step_rtg_t = torch.tensor(rtg_list, dtype=torch.float32, device=device)
            step_rtg_t = step_rtg_t.unsqueeze(0).unsqueeze(-1)

        return {
            "epi_states":     epi_s,
            "epi_opp_acts":   epi_a,
            "step_states":    step_states_t,
            "step_self_acts": step_sa_t,
            "step_opp_acts":  step_oa_t,
            "step_rtgs":      step_rtg_t,
            "step_timesteps": step_ts_t,
        }


# ─────────────────────────────────────────────────────────────────
# Quick model sanity check
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, H, L = 2, H_EPI, 5
    model = OMISModel()
    print(f"OMISModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    epi_states    = torch.randn(B, H, STATE_DIM)
    epi_opp_acts  = torch.randint(0, 324, (B, H))
    step_states   = torch.randn(B, L, STATE_DIM)
    step_self_acts = torch.randint(0, 18, (B, L - 1))
    step_opp_acts  = torch.randint(0, 324, (B, L - 1))
    step_rtgs      = torch.randn(B, L - 1, 1)
    step_timesteps = torch.arange(L).unsqueeze(0).expand(B, -1)

    actor_logits, imitator_logits, critic_values = model(
        epi_states, epi_opp_acts,
        step_states, step_self_acts, step_opp_acts, step_rtgs, step_timesteps,
    )

    print(f"actor_logits shape   : {actor_logits.shape}")     # (B, H+L, 18)
    print(f"imitator_logits shape: {imitator_logits.shape}")  # (B, H+L, 324)
    print(f"critic_values shape  : {critic_values.shape}")    # (B, H+L, 1)
    print("OMISModel forward pass OK")
