"""
brAgent.py
==========
PPO Best Response (BR) agent trained against a fixed scripted opponent policy.
For each policy π⁻¹,k in Pi_train, we train a BR agent BR(π⁻¹,k) using PPO.
These BRs are then used to generate pretraining data for the OMIS Transformer.

Architecture (Appendix H.2):
  - 3 linear layers, 32 hidden nodes
  - Actor LR = 5e-4, Critic LR = 5e-4
  - PPO clip = 0.2, gradient clip = 5.0
  - Discount γ = 1.0
  - Batch size = 4096, epochs per update = 10
  - Total training episodes = 50,000

The BR is trained as the PREY agent (self-agent, index 2) while
Predator1 and Predator2 follow the fixed scripted policy π⁻¹,k.

Input: 128-dim VoxelEncoder output (state representation)
Output: logits over 30 actions (3 move * 5 turn_bins * 2 attack)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


# ─────────────────────────────────────────────────────────────────
# Hyperparameters (Appendix H.2)
# ─────────────────────────────────────────────────────────────────

STATE_DIM         = 128
N_ACTIONS         = 30      # 3 move * 5 turn_bins * 2 attack (flat)
HIDDEN_DIM        = 32
N_LAYERS          = 3
LR_ACTOR          = 5e-4
LR_CRITIC         = 5e-4
PPO_CLIP          = 0.2
GRAD_CLIP         = 5.0
GAMMA             = 1.0     # discount factor for BR training
BATCH_SIZE        = 4096
N_EPOCHS          = 10      # PPO update epochs per batch
TOTAL_EPISODES    = 50
GAE_LAMBDA        = 0.95    # for advantage estimation


# ─────────────────────────────────────────────────────────────────
# Actor-Critic Network
# ─────────────────────────────────────────────────────────────────

def _build_mlp(in_dim, hidden_dim, n_layers, out_dim, activation=nn.ReLU):
    """Build an MLP with n_layers hidden layers."""
    layers = []
    dim = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(activation())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class BRActorCritic(nn.Module):
    """
    Shared-trunk actor-critic for PPO Best Response training.
    Actor:  outputs logits over 30 flat actions
    Critic: outputs scalar state value
    """

    def __init__(
        self,
        state_dim  = STATE_DIM,
        n_actions  = N_ACTIONS,
        hidden_dim = HIDDEN_DIM,
        n_layers   = N_LAYERS,
    ):
        super().__init__()
        # Shared trunk
        trunk_layers = []
        dim = state_dim
        for _ in range(n_layers - 1):
            trunk_layers += [nn.Linear(dim, hidden_dim), nn.ReLU()]
            dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        # Actor head
        self.actor_head  = nn.Linear(dim, n_actions)
        # Critic head
        self.critic_head = nn.Linear(dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        """state: (B, state_dim) → action_logits (B, n_actions), value (B, 1)"""
        x      = self.trunk(state)
        logits = self.actor_head(x)
        value  = self.critic_head(x)
        return logits, value

    def get_action_and_value(self, state):
        """Sample action and return (action, log_prob, value, entropy)."""
        logits, value = self.forward(state)
        dist          = torch.distributions.Categorical(logits=logits)
        action        = dist.sample()
        log_prob      = dist.log_prob(action)
        entropy       = dist.entropy()
        return action, log_prob, value.squeeze(-1), entropy

    def evaluate_actions(self, state, actions):
        """Evaluate log_prob and value for given (state, action) pairs."""
        logits, value = self.forward(state)
        dist          = torch.distributions.Categorical(logits=logits)
        log_prob      = dist.log_prob(actions)
        entropy       = dist.entropy()
        return log_prob, value.squeeze(-1), entropy

    @torch.no_grad()
    def act(self, state_np):
        """
        Greedy action selection for deployment.
        state_np: numpy array (state_dim,)
        Returns: flat action index (int)
        """
        state  = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        logits, _ = self.forward(state)
        action = torch.argmax(logits, dim=-1)
        return action.item()

    @torch.no_grad()
    def sample_action(self, state_np):
        """
        Stochastic action for data collection.
        Returns: (flat_action_int, log_prob_float)
        """
        state  = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        action, log_prob, _, _ = self.get_action_and_value(state)
        return action.item(), log_prob.item()


# ─────────────────────────────────────────────────────────────────
# PPO Rollout Buffer
# ─────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores a single PPO rollout for the self-agent (prey).
    Computes GAE advantages after the rollout is complete.
    """

    def __init__(self, capacity=BATCH_SIZE):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def is_full(self):
        return len(self.states) >= self.capacity

    def compute_returns_and_advantages(self, last_value=0.0, gamma=GAMMA, lam=GAE_LAMBDA):
        """
        Compute discounted returns and GAE advantages.
        last_value: bootstrap value for the last state (0 if terminal).
        """
        n        = len(self.rewards)
        returns  = np.zeros(n, dtype=np.float32)
        advs     = np.zeros(n, dtype=np.float32)
        gae      = 0.0
        next_val = last_value

        for t in reversed(range(n)):
            delta   = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae     = delta + gamma * lam * (1 - self.dones[t]) * gae
            advs[t] = gae
            next_val = self.values[t]

        returns = advs + np.array(self.values, dtype=np.float32)
        return returns, advs

    def get_tensors(self, last_value=0.0):
        """Return all data as PyTorch tensors."""
        returns, advs = self.compute_returns_and_advantages(last_value)

        states    = torch.tensor(np.array(self.states),    dtype=torch.float32)
        actions   = torch.tensor(np.array(self.actions),   dtype=torch.long)
        log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32)
        returns   = torch.tensor(returns,                  dtype=torch.float32)
        advs      = torch.tensor(advs,                     dtype=torch.float32)

        # Normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        return states, actions, log_probs, returns, advs


# ─────────────────────────────────────────────────────────────────
# PPO Trainer
# ─────────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    PPO trainer for the Best Response agent.
    Trains the prey agent against a fixed scripted predator policy.
    """

    def __init__(
        self,
        policy_idx,             # index k of the scripted opponent policy
        state_dim  = STATE_DIM,
        n_actions  = N_ACTIONS,
        hidden_dim = HIDDEN_DIM,
        n_layers   = N_LAYERS,
        lr_actor   = LR_ACTOR,
        lr_critic  = LR_CRITIC,
        ppo_clip   = PPO_CLIP,
        grad_clip  = GRAD_CLIP,
        gamma      = GAMMA,
        batch_size = BATCH_SIZE,
        n_epochs   = N_EPOCHS,
        device     = "cpu",
    ):
        self.policy_idx = policy_idx
        self.ppo_clip   = ppo_clip
        self.grad_clip  = grad_clip
        self.gamma      = gamma
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.device     = device

        self.net = BRActorCritic(state_dim, n_actions, hidden_dim, n_layers).to(device)

        # Separate optimizers for actor and critic (Appendix H.2)
        self.opt_actor  = optim.Adam(
            list(self.net.trunk.parameters()) + list(self.net.actor_head.parameters()),
            lr=lr_actor
        )
        self.opt_critic = optim.Adam(
            list(self.net.trunk.parameters()) + list(self.net.critic_head.parameters()),
            lr=lr_critic
        )

        self.buffer = RolloutBuffer(capacity=batch_size)
        self.total_updates = 0

    def select_action(self, state_np):
        """
        Select action during rollout collection.
        Returns (action_int, log_prob_float, value_float)
        """
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value, _ = self.net.get_action_and_value(state)
        return action.item(), log_prob.item(), value.item()

    def store(self, state, action, log_prob, reward, value, done):
        """Store a transition in the rollout buffer."""
        self.buffer.add(state, action, log_prob, reward, value, done)

    def update(self, last_value=0.0):
        """
        Run PPO update on the current buffer contents.
        Called when buffer is full or episode ends.
        Returns dict of losses for logging.
        """
        states, actions, old_log_probs, returns, advs = self.buffer.get_tensors(last_value)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns     = returns.to(self.device)
        advs        = advs.to(self.device)

        n          = len(states)
        actor_losses  = []
        critic_losses = []
        entropy_losses = []

        for _ in range(self.n_epochs):
            # Shuffle
            idx = torch.randperm(n)
            for start in range(0, n, 256):
                batch_idx = idx[start: start + 256]
                s  = states[batch_idx]
                a  = actions[batch_idx]
                lp = old_log_probs[batch_idx].detach()
                r  = returns[batch_idx].detach()
                adv = advs[batch_idx].detach()

                new_log_prob, value_pred, entropy = self.net.evaluate_actions(s, a)

                # PPO ratio
                ratio      = torch.exp(new_log_prob - lp)
                surr1      = ratio * adv
                surr2      = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(value_pred, r)
                entropy_loss = -entropy.mean()

                # Combined update — fixes inplace modification error on shared trunk
                total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.opt_actor.step()
                self.opt_critic.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        self.buffer.reset()
        self.total_updates += 1

        return {
            "actor_loss":  np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
        }

    def save(self, path):
        """Save model weights to path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "net":           self.net.state_dict(),
            "opt_actor":     self.opt_actor.state_dict(),
            "opt_critic":    self.opt_critic.state_dict(),
            "total_updates": self.total_updates,
            "policy_idx":    self.policy_idx,
        }, path)
        print(f"[BR] Saved checkpoint → {path}")

    def load(self, path):
        """Load model weights from path."""
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
        self.total_updates = ckpt.get("total_updates", 0)
        print(f"[BR] Loaded checkpoint ← {path}")

    @staticmethod
    def load_for_inference(path, device="cpu"):
        """Load a trained BR network for data collection only."""
        ckpt = torch.load(path, map_location=device)
        net  = BRActorCritic().to(device)
        net.load_state_dict(ckpt["net"])
        net.eval()
        return net
