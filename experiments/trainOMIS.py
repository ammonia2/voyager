"""
trainOMIS.py
============
Phase 2 of OMIS pipeline: supervised in-context pretraining
for the shared dual-predator OMIS model.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.omisAgent import OMISModel
from utils.dataCollector import DataCollector, PretrainDataset
from utils.logs import MARLLogger
from envs.malmoEnvOmis import MalmoEnv


MISSION_XML = os.path.join(os.path.dirname(__file__), "..", "configs", "missionPredatorPrey.xml")
BR_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "br")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pretrain")
OMIS_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "omis")

TOTAL_STEPS = 50
BATCH_SIZE = 64
N_EPOCHS = 10
LR = 6e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 0.5
WARMUP_STEPS = 10000

W_ACTOR = 1.0
W_IMITATOR_PREY = 0.8
W_IMITATOR_PRED = 0.3
W_CRITIC = 0.5

LOG_INTERVAL = 1
SAVE_INTERVAL = 500
N_COLLECT_EPISODES = 20


def get_lr(step, warmup_steps=WARMUP_STEPS, base_lr=LR):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def train_step(model, batch, optimizer, device):
    model.train()

    epi_states = batch["epi_states"].to(device)
    epi_pred_opp_acts = batch["epi_pred_opp_acts"].to(device)
    epi_prey_opp_acts = batch["epi_prey_opp_acts"].to(device)

    step_states = batch["step_states"].to(device)
    step_self_acts = batch["step_self_acts"].to(device)
    step_pred_opp_acts = batch["step_pred_opp_acts"].to(device)
    step_prey_opp_acts = batch["step_prey_opp_acts"].to(device)
    step_rtgs = batch["step_rtgs"].to(device)
    step_timesteps = batch["step_timesteps"].to(device)

    label_self_act = batch["label_self_act"].to(device)
    label_pred_opp_act = batch["label_pred_opp_act"].to(device)
    label_prey_opp_act = batch["label_prey_opp_act"].to(device)
    label_rtg = batch["label_rtg"].to(device)

    if step_rtgs.dim() == 2:
        step_rtgs = step_rtgs.unsqueeze(-1)

    actor_logits, imitator_pred_logits, imitator_prey_logits, critic_values = model(
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

    actor_pred = actor_logits[:, -1, :]
    pred_opp_pred = imitator_pred_logits[:, -1, :]
    prey_opp_pred = imitator_prey_logits[:, -1, :]
    critic_pred = critic_values[:, -1, 0]

    l_actor = F.cross_entropy(actor_pred, label_self_act)
    l_imitator_pred = F.cross_entropy(pred_opp_pred, label_pred_opp_act)
    l_imitator_prey = F.cross_entropy(prey_opp_pred, label_prey_opp_act)
    l_critic = F.mse_loss(critic_pred, label_rtg)

    loss = (
        W_ACTOR * l_actor
        + W_IMITATOR_PREY * l_imitator_prey
        + W_IMITATOR_PRED * l_imitator_pred
        + W_CRITIC * l_critic
    )

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    return {
        "loss": loss.item(),
        "actor_loss": l_actor.item(),
        "imitator_pred_loss": l_imitator_pred.item(),
        "imitator_prey_loss": l_imitator_prey.item(),
        "critic_loss": l_critic.item(),
    }


@torch.no_grad()
def evaluate(model, dataloader, device, n_batches=20):
    model.eval()
    total = 0

    actor_correct = 0
    pred_opp_correct = 0
    prey_opp_correct = 0
    critic_mse_sum = 0.0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break

        epi_states = batch["epi_states"].to(device)
        epi_pred_opp_acts = batch["epi_pred_opp_acts"].to(device)
        epi_prey_opp_acts = batch["epi_prey_opp_acts"].to(device)

        step_states = batch["step_states"].to(device)
        step_self_acts = batch["step_self_acts"].to(device)
        step_pred_opp_acts = batch["step_pred_opp_acts"].to(device)
        step_prey_opp_acts = batch["step_prey_opp_acts"].to(device)
        step_rtgs = batch["step_rtgs"].to(device)
        step_timesteps = batch["step_timesteps"].to(device)

        label_self_act = batch["label_self_act"].to(device)
        label_pred_opp_act = batch["label_pred_opp_act"].to(device)
        label_prey_opp_act = batch["label_prey_opp_act"].to(device)
        label_rtg = batch["label_rtg"].to(device)

        if step_rtgs.dim() == 2:
            step_rtgs = step_rtgs.unsqueeze(-1)

        actor_logits, imitator_pred_logits, imitator_prey_logits, critic_values = model(
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

        actor_pred = actor_logits[:, -1, :]
        pred_opp_pred = imitator_pred_logits[:, -1, :]
        prey_opp_pred = imitator_prey_logits[:, -1, :]
        critic_pred = critic_values[:, -1, 0]

        bsz = label_self_act.size(0)
        total += bsz

        actor_correct += (actor_pred.argmax(dim=-1) == label_self_act).sum().item()
        pred_opp_correct += (pred_opp_pred.argmax(dim=-1) == label_pred_opp_act).sum().item()
        prey_opp_correct += (prey_opp_pred.argmax(dim=-1) == label_prey_opp_act).sum().item()
        critic_mse_sum += F.mse_loss(critic_pred, label_rtg).item() * bsz

    if total == 0:
        return {}

    return {
        "actor_acc": actor_correct / total,
        "pred_opp_acc": pred_opp_correct / total,
        "prey_opp_acc": prey_opp_correct / total,
        "critic_mse": critic_mse_sum / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Pretrain shared OMIS Transformer")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--collect", action="store_true", help="Collect pretraining data first")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    os.makedirs(OMIS_CKPT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.collect:
        print("\n[trainOMIS] Collecting pretraining data...", flush=True)
        env = MalmoEnv(MISSION_XML)
        collector = DataCollector(env, BR_CKPT_DIR, DATA_DIR, device=str(device))
        collector.collect(n_episodes_per_policy=N_COLLECT_EPISODES)
        print("[trainOMIS] Data collection done.\n", flush=True)

    print("[trainOMIS] Loading pretraining dataset...", flush=True)
    dataset = PretrainDataset(DATA_DIR)
    if len(dataset) == 0:
        print("[trainOMIS] ERROR: No pretraining data found.", flush=True)
        print("Run with --collect, or run trainBR.py first.", flush=True)
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=PretrainDataset.collate_fn,
        drop_last=True,
    )

    model = OMISModel().to(device)
    print(f"[trainOMIS] Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    start_step = 0
    ckpt_path = os.path.join(OMIS_CKPT_DIR, "omis_pretrained.pt")

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"[trainOMIS] Resumed from step {start_step}", flush=True)

    logger = MARLLogger(algo_name="OMIS_pretraining_predator_shared", log_interval=LOG_INTERVAL, seed=0)

    print(f"\n[trainOMIS] Starting pretraining for {args.total_steps} steps...", flush=True)
    data_iter = iter(dataloader)

    for step in range(start_step, args.total_steps):
        lr_now = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        step_losses = {
            "loss": 0.0,
            "actor_loss": 0.0,
            "imitator_pred_loss": 0.0,
            "imitator_prey_loss": 0.0,
            "critic_loss": 0.0,
        }

        for _ in range(N_EPOCHS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            losses = train_step(model, batch, optimizer, device)
            for k, v in losses.items():
                step_losses[k] += v / N_EPOCHS

        print(
            f"  Step {step + 1:4d} | Loss {step_losses['loss']:.4f} | "
            f"Actor {step_losses['actor_loss']:.4f} | "
            f"ImPred {step_losses['imitator_pred_loss']:.4f} | "
            f"ImPrey {step_losses['imitator_prey_loss']:.4f} | "
            f"Critic {step_losses['critic_loss']:.4f}",
            flush=True,
        )

        if (step + 1) % LOG_INTERVAL == 0:
            eval_metrics = evaluate(model, dataloader, device)
            if eval_metrics:
                print(
                    f"  [Eval] Step {step + 1:4d} | "
                    f"Actor {eval_metrics['actor_acc']*100:.1f}% | "
                    f"PredOpp {eval_metrics['pred_opp_acc']*100:.1f}% | "
                    f"PreyOpp {eval_metrics['prey_opp_acc']*100:.1f}% | "
                    f"CriticMSE {eval_metrics['critic_mse']:.4f}",
                    flush=True,
                )
                logger.log_eval(
                    episode=step + 1,
                    episode_return=0.0,
                    pred_opp_acc=eval_metrics["pred_opp_acc"],
                    prey_opp_acc=eval_metrics["prey_opp_acc"],
                    value_mse=eval_metrics["critic_mse"],
                    opponent_type="seen",
                )

        if (step + 1) % SAVE_INTERVAL == 0 or step == args.total_steps - 1:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                },
                ckpt_path,
            )
            print(f"  [trainOMIS] Checkpoint saved at step {step + 1} -> {ckpt_path}", flush=True)

    logger.print_final_summary()
    print(f"\n[trainOMIS] Pretraining complete. Model saved to: {ckpt_path}", flush=True)
    print("Next step: run experiments/evalOMIS.py", flush=True)


if __name__ == "__main__":
    main()
