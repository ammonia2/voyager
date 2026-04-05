"""
trainOMIS.py
============
Phase 2 of OMIS pipeline: ICL-based pretraining of the OMIS Transformer.

Trains the three in-context components jointly using supervised learning:
  πθ  (actor)    — cross-entropy loss on self-agent action labels  (weight 1.0)
  µϕ  (imitator) — cross-entropy loss on opponent action labels    (weight 0.8)
  Vω  (critic)   — MSE loss on RTG labels                         (weight 0.5)

Total loss = 1.0 * L_actor + 0.8 * L_imitator + 0.5 * L_critic

Follows Algorithm 1 lines 5-9 and Equations (3)-(5).

Usage:
    conda activate marl-malmo
    # First run trainBR.py and collect data, then:
    python experiments/trainOMIS.py
    python experiments/trainOMIS.py --resume

Hyperparameters (Appendix H.2):
    Training steps     : 4000
    Batch size         : 64
    Epochs per step    : 10
    LR (AdamW)         : 6e-4
    Weight decay       : 1e-4
    Gradient clip      : 0.5
    Warmup steps       : 10000
    Discount γ         : 1.0  (already applied in data collection)
    Reward scaling     : 100  (already applied in data collection)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.omisAgent    import OMISModel, H_EPI, B_SEQ
from utils.dataCollector import DataCollector, PretrainDataset
from utils.logs           import MARLLogger
from envs.malmoEnvOmis       import MalmoEnv
from models.voxelEncoder import VoxelEncoder

# ─────────────────────────────────────────────────────────────────
# Config (Appendix H.2)
# ─────────────────────────────────────────────────────────────────

MISSION_XML   = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
BR_CKPT_DIR   = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "br")
DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data", "pretrain")
OMIS_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "omis")

TOTAL_STEPS     = 50
BATCH_SIZE      = 64
N_EPOCHS        = 10      # epochs per training step
LR              = 6e-4
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 0.5
WARMUP_STEPS    = 10000   # linear LR warmup

# Loss weighting coefficients (Appendix H.2)
W_ACTOR    = 1.0
W_IMITATOR = 0.8
W_CRITIC   = 0.5

LOG_INTERVAL  = 1
SAVE_INTERVAL = 500

# Data collection params
N_COLLECT_EPISODES = 20   # episodes per policy for data collection


# ─────────────────────────────────────────────────────────────────
# Learning rate scheduler with warmup
# ─────────────────────────────────────────────────────────────────

def get_lr(step, warmup_steps=WARMUP_STEPS, base_lr=LR):
    """Linear warmup then constant LR."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


# ─────────────────────────────────────────────────────────────────
# One training step on a batch
# ─────────────────────────────────────────────────────────────────

def train_step(model, batch, optimizer, device):
    """
    Run one supervised update on a batch of PretrainSamples.
    Returns dict of loss values.
    """
    model.train()

    # Move all tensors to device
    epi_states     = batch["epi_states"].to(device)       # (B, H, 128)
    epi_opp_acts   = batch["epi_opp_acts"].to(device)     # (B, H)
    step_states    = batch["step_states"].to(device)      # (B, L, 128)
    step_self_acts = batch["step_self_acts"].to(device)   # (B, L-1)
    step_opp_acts  = batch["step_opp_acts"].to(device)    # (B, L-1)
    step_rtgs      = batch["step_rtgs"].to(device)        # (B, L-1)
    step_timesteps = batch["step_timesteps"].to(device)   # (B, L)
    label_self_act = batch["label_self_act"].to(device)   # (B,)
    label_opp_act  = batch["label_opp_act"].to(device)    # (B,)
    label_rtg      = batch["label_rtg"].to(device)        # (B,)

    # Handle variable-length step_rtgs: needs shape (B, L-1, 1)
    if step_rtgs.dim() == 2:
        step_rtgs = step_rtgs.unsqueeze(-1)

    # Forward pass
    actor_logits, imitator_logits, critic_values = model(
        epi_states, epi_opp_acts,
        step_states, step_self_acts, step_opp_acts, step_rtgs, step_timesteps,
    )

    # Predictions are at all state positions; we want the LAST position (query state s_t)
    actor_pred    = actor_logits[:, -1, :]     # (B, n_self_actions)
    imitator_pred = imitator_logits[:, -1, :]  # (B, n_joint_opp)
    critic_pred   = critic_values[:, -1, 0]    # (B,)

    # ── Losses (Equations 3, 4, 5) ───────────────────────────────
    # Eq. 3: actor cross-entropy
    l_actor = F.cross_entropy(actor_pred, label_self_act)

    # Eq. 4: imitator cross-entropy
    l_imitator = F.cross_entropy(imitator_pred, label_opp_act)

    # Eq. 5: critic MSE
    l_critic = F.mse_loss(critic_pred, label_rtg)

    # Weighted total loss
    loss = W_ACTOR * l_actor + W_IMITATOR * l_imitator + W_CRITIC * l_critic

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    return {
        "loss":          loss.item(),
        "actor_loss":    l_actor.item(),
        "imitator_loss": l_imitator.item(),
        "critic_loss":   l_critic.item(),
    }


# ─────────────────────────────────────────────────────────────────
# Evaluation on held-out batch
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, n_batches=20):
    """
    Evaluate actor accuracy and critic MSE on a subset of the dataset.
    Returns dict of metrics.
    """
    model.eval()
    actor_correct  = 0
    imitator_correct = 0
    critic_mse_sum = 0.0
    total          = 0

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        epi_states     = batch["epi_states"].to(device)
        epi_opp_acts   = batch["epi_opp_acts"].to(device)
        step_states    = batch["step_states"].to(device)
        step_self_acts = batch["step_self_acts"].to(device)
        step_opp_acts  = batch["step_opp_acts"].to(device)
        step_rtgs      = batch["step_rtgs"].to(device)
        step_timesteps = batch["step_timesteps"].to(device)
        label_self_act = batch["label_self_act"].to(device)
        label_opp_act  = batch["label_opp_act"].to(device)
        label_rtg      = batch["label_rtg"].to(device)

        if step_rtgs.dim() == 2:
            step_rtgs = step_rtgs.unsqueeze(-1)

        actor_logits, imitator_logits, critic_values = model(
            epi_states, epi_opp_acts,
            step_states, step_self_acts, step_opp_acts, step_rtgs, step_timesteps,
        )

        actor_pred    = actor_logits[:, -1, :]
        imitator_pred = imitator_logits[:, -1, :]
        critic_pred   = critic_values[:, -1, 0]

        B = label_self_act.size(0)
        actor_correct    += (actor_pred.argmax(dim=-1) == label_self_act).sum().item()
        imitator_correct += (imitator_pred.argmax(dim=-1) == label_opp_act).sum().item()
        critic_mse_sum   += F.mse_loss(critic_pred, label_rtg).item() * B
        total            += B

    if total == 0:
        return {}

    return {
        "actor_acc":    actor_correct / total,
        "imitator_acc": imitator_correct / total,
        "critic_mse":   critic_mse_sum / total,
    }


# ─────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pretrain OMIS Transformer")
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--collect",     action="store_true",
                        help="Collect pretraining data before training")
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    os.makedirs(OMIS_CKPT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Optionally collect data ──────────────────────────────────
    if args.collect:
        print("\n[trainOMIS] Collecting pretraining data...", flush=True)
        env           = MalmoEnv(MISSION_XML)
        voxel_encoder = VoxelEncoder().to(device)
        voxel_encoder.eval()
        collector = DataCollector(env, BR_CKPT_DIR, DATA_DIR, device=str(device))
        collector.collect(n_episodes_per_policy=N_COLLECT_EPISODES)
        print("[trainOMIS] Data collection done.\n")

    # ── Load dataset ─────────────────────────────────────────────
    print("[trainOMIS] Loading pretraining dataset...")
    dataset = PretrainDataset(DATA_DIR)

    if len(dataset) == 0:
        print("[trainOMIS] ERROR: No pretraining data found.")
        print("  Run with --collect, or run trainBR.py first.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,
        collate_fn  = PretrainDataset.collate_fn,
        drop_last   = True,
    )

    # ── Build model ───────────────────────────────────────────────
    model = OMISModel().to(device)
    print(f"[trainOMIS] Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = LR,
        weight_decay = WEIGHT_DECAY,
    )

    start_step = 0
    ckpt_path  = os.path.join(OMIS_CKPT_DIR, "omis_pretrained.pt")

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"[trainOMIS] Resumed from step {start_step}", flush=True)

    logger = MARLLogger(
        algo_name    = "OMIS_pretraining",
        log_interval = LOG_INTERVAL,
        seed         = 0,
    )

    # ── Training loop ─────────────────────────────────────────────
    print(f"\n[trainOMIS] Starting pretraining for {args.total_steps} steps...", flush=True)
    data_iter = iter(dataloader)

    for step in range(start_step, args.total_steps):

        # Adjust learning rate (warmup)
        lr_now = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Run N_EPOCHS mini-updates per step (Appendix H.2)
        step_losses = {"loss": 0, "actor_loss": 0, "imitator_loss": 0, "critic_loss": 0}

        for _ in range(N_EPOCHS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch     = next(data_iter)

            losses = train_step(model, batch, optimizer, device)
            for k, v in losses.items():
                step_losses[k] += v / N_EPOCHS

        # Log
        # logger.log_step(
        #     step           = step,
        #     loss           = step_losses["loss"],
        #     actor_loss     = step_losses["actor_loss"],
        #     imitator_loss  = step_losses["imitator_loss"],
        #     critic_loss    = step_losses["critic_loss"],
        #     lr             = lr_now,
        # )
        
        # Print progress every step
        if (step + 1) % 1 == 0:
            print(
                f"  Step {step+1:4d} | Loss: {step_losses['loss']:.4f} | "
                f"Actor: {step_losses['actor_loss']:.4f} | "
                f"Critic: {step_losses['critic_loss']:.4f}",
                flush=True
            )

        # Periodic evaluation
        if (step + 1) % LOG_INTERVAL == 0:
            eval_metrics = evaluate(model, dataloader, device)
            if eval_metrics:
                print(
                    f"  [Eval] Step {step+1:4d} | "
                    f"Actor Acc: {eval_metrics['actor_acc']*100:.1f}% | "
                    f"Imitator Acc: {eval_metrics['imitator_acc']*100:.1f}% | "
                    f"Critic MSE: {eval_metrics['critic_mse']:.4f}",
                    flush=True
                )
                # Log to MARLLogger for eval tracking
                logger.log_eval(
                    episode        = step + 1,
                    episode_return = 0.0,
                    opponent_acc   = eval_metrics["imitator_acc"],
                    value_mse      = eval_metrics["critic_mse"],
                    opponent_type  = "seen",
                )

        # Save checkpoint
        if (step + 1) % SAVE_INTERVAL == 0 or step == args.total_steps - 1:
            torch.save({
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step":      step + 1,
            }, ckpt_path)
            print(f"  [trainOMIS] Checkpoint saved at step {step+1} → {ckpt_path}", flush=True)

    logger.print_final_summary()
    print(f"\n[trainOMIS] Pretraining complete. Model saved to: {ckpt_path}", flush=True)
    print("Next step: run experiments/evalOMIS.py", flush=True)


if __name__ == "__main__":
    main()
