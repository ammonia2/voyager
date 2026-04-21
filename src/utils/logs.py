"""
log.py - Standard Evaluation Logger for MARL Opponent Modeling
===============================================================
Metrics sourced from all three base papers:
  - OMIS     (Jing et al., NeurIPS 2024)
  - OEOM     (Jing et al., AAAI 2025)
  - MARLeOM  (Li et al., ECAI 2024)

Output is printed to console; optional JSONL file logging is supported.
"""

import json
import os
import time
from collections import defaultdict


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _mean(values):
    return sum(values) / len(values) if values else 0.0

def _std(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5

def _timestamp():
    return time.strftime("%H:%M:%S")

def _divider(char="─", width=60):
    print(char * width)


# ──────────────────────────────────────────────────────────────
# MetricBuffer  —  stores raw values per episode/step
# ──────────────────────────────────────────────────────────────

class MetricBuffer:
    """
    Accumulates raw per-episode or per-step metric values.
    Call .record() each episode, then .summary() to get stats.
    Call .reset() to start a new logging window.
    """

    def __init__(self):
        self._data = defaultdict(list)

    def record(self, **kwargs):
        """
        Record one set of values for this episode/step.

        Example:
            buf.record(episode_return=12.5, win=1, opponent_action_acc=0.63)
        """
        for key, val in kwargs.items():
            self._data[key].append(float(val))

    def summary(self):
        """Return {metric: (mean, std)} for all recorded metrics."""
        return {k: (_mean(v), _std(v)) for k, v in self._data.items()}

    def reset(self):
        self._data = defaultdict(list)

    def __len__(self):
        # Number of episodes recorded (based on first key)
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))


# ──────────────────────────────────────────────────────────────
# MARLLogger  —  main interface
# ──────────────────────────────────────────────────────────────

class MARLLogger:
    """
    Central logger for MARL opponent modeling experiments.

    Usage
    -----
        logger = MARLLogger(algo_name="OMIS", log_interval=100)

        # Inside your training loop:
        logger.log_episode(
            episode        = ep,
            episode_return = total_reward,
            win            = int(team_won),          # 0 or 1
            opponent_type  = "unseen",               # "seen" | "unseen"
        )

        # Inside your training step (optional, e.g. for loss values):
        logger.log_step(
            step                    = global_step,
            actor_loss              = loss_actor,
            opponent_imitator_loss  = loss_imitator,   # OMIS
            critic_loss             = loss_critic,
        )

        # At the end of a logging window (every log_interval episodes):
        # Called automatically inside log_episode — or call manually:
        logger.print_episode_summary(episode=ep)

        # At eval time against seen/unseen opponents:
        logger.log_eval(
            episode        = ep,
            episode_return = eval_return,
            win            = int(team_won),
            opponent_acc   = pred_accuracy,          # OMIS / MARLeOM
            value_mse      = critic_mse,             # OMIS
            opponent_type  = "unseen",
        )
    """

    def __init__(self, algo_name="MARL", log_interval=100, seed=0, log_file=None):
        self.algo_name    = algo_name
        self.log_interval = log_interval
        self.seed         = seed
        self.log_file     = log_file

        if self.log_file is not None:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Buffers
        self._train_buf   = MetricBuffer()   # rolling training window
        self._eval_seen   = MetricBuffer()   # eval vs seen opponents
        self._eval_unseen = MetricBuffer()   # eval vs unseen opponents

        # Totals for win-rate across full run
        self._total_episodes = 0
        self._total_wins     = 0

        _divider("═")
        print(f"  [{_timestamp()}]  Logger started")
        print(f"  Algorithm : {self.algo_name}")
        print(f"  Seed      : {self.seed}")
        print(f"  Log every : {self.log_interval} episodes")
        _divider("═")

    def _write_record(self, record_type, **payload):
        """Append one JSONL record if file logging is enabled."""
        if self.log_file is None:
            return
        record = {
            "timestamp": _timestamp(),
            "algorithm": self.algo_name,
            "record_type": record_type,
        }
        record.update(payload)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ── Training episode logging ──────────────────────────────

    def log_episode(self, episode, episode_return, win=None, opponent_type=None, **extra):
        """
        Record one completed training episode.

        Parameters
        ----------
        episode        : int   — episode number (1-indexed)
        episode_return : float — total cumulative reward this episode
        win            : int   — 1 if agent/team won, 0 otherwise (optional)
        opponent_type  : str   — "seen" | "unseen" | None
        **extra        : any additional scalar metrics to record
        """
        payload = {"episode_return": episode_return}
        if win is not None:
            payload["win"] = win
            self._total_wins += int(win)
        if opponent_type is not None:
            payload["is_unseen"] = int(opponent_type == "unseen")
        payload.update(extra)

        self._train_buf.record(**payload)
        self._total_episodes += 1
        self._write_record("episode", episode=int(episode), **payload)

        if episode % self.log_interval == 0:
            self.print_episode_summary(episode)
            self._train_buf.reset()

    def log_step(self, step, **losses):
        """
        Optionally log per-step training losses.

        Parameters
        ----------
        step    : int   — global training step
        **losses: named loss values (actor_loss, critic_loss, etc.)
        """
        parts = [f"{k}={v:.4f}" for k, v in losses.items()]
        print(f"  [{_timestamp()}]  Step {step:>8d}  |  " + "  ".join(parts))
        self._write_record("step", step=int(step), **losses)

    # ── Evaluation logging ────────────────────────────────────

    def log_eval(self, episode, episode_return, win=None,
                 opponent_acc=None, value_mse=None, opponent_type="unseen", **extra):
        """
        Record one evaluation episode result.

        Parameters
        ----------
        episode        : int   — current training episode (context)
        episode_return : float — eval episode return
        win            : int   — 1/0 win flag (optional)
        opponent_acc   : float — opponent action prediction accuracy 0-1 (OMIS / MARLeOM)
        value_mse      : float — critic value estimation MSE (OMIS)
        opponent_type  : str   — "seen" | "unseen"
        """
        payload = {"episode_return": episode_return}
        if win is not None:
            payload["win"] = win
        if opponent_acc is not None:
            payload["opponent_acc"] = opponent_acc
        if value_mse is not None:
            payload["value_mse"] = value_mse
        payload.update(extra)

        if opponent_type == "seen":
            self._eval_seen.record(**payload)
        else:
            self._eval_unseen.record(**payload)

        self._write_record("eval", episode=int(episode), opponent_type=opponent_type, **payload)

    def print_eval_summary(self, episode, n_seen_episodes=None, n_unseen_episodes=None):
        """
        Print a formatted evaluation summary.
        Call this after all eval episodes for a checkpoint are done.

        Parameters
        ----------
        episode           : int — training episode at which eval was run
        n_seen_episodes   : int — how many eval episodes used seen opponents
        n_unseen_episodes : int — how many eval episodes used unseen opponents
        """
        _divider()
        print(f"  [{_timestamp()}]  EVAL SUMMARY  |  {self.algo_name}  |  Episode {episode}")
        _divider()

        for label, buf, n in [
            ("SEEN opponents",   self._eval_seen,   n_seen_episodes),
            ("UNSEEN opponents", self._eval_unseen, n_unseen_episodes),
        ]:
            stats = buf.summary()
            if not stats:
                continue
            n_eps = n or len(buf)
            print(f"\n  {label}  ({n_eps} episodes)")

            # ── Core metrics (consistent across all 3 papers) ──
            if "episode_return" in stats:
                m, s = stats["episode_return"]
                print(f"    Avg Episode Return   : {m:+.3f}  ±  {s:.3f}")

            if "win" in stats:
                m, s = stats["win"]
                print(f"    Win Rate             : {m * 100:.1f}%  ±  {s * 100:.1f}%")

            # ── OMIS / MARLeOM: opponent action prediction ──
            if "opponent_acc" in stats:
                m, s = stats["opponent_acc"]
                print(f"    Opponent Action Acc  : {m * 100:.1f}%  ±  {s * 100:.1f}%")

            # ── OMIS: critic value estimation ──
            if "value_mse" in stats:
                m, s = stats["value_mse"]
                print(f"    Value Estimation MSE : {m:.5f}  ±  {s:.5f}")

            # ── Any extra metrics ──
            skip = {"episode_return", "win", "opponent_acc", "value_mse"}
            for k, (m, s) in stats.items():
                if k not in skip:
                    print(f"    {k:<22} : {m:.4f}  ±  {s:.4f}")

        # ── Generalization gap (OMIS / OEOM) ──
        seen_stats   = self._eval_seen.summary()
        unseen_stats = self._eval_unseen.summary()
        if "episode_return" in seen_stats and "episode_return" in unseen_stats:
            gap = seen_stats["episode_return"][0] - unseen_stats["episode_return"][0]
            print(f"\n  Generalization Gap (seen - unseen return): {gap:+.3f}")

        _divider()

        # Reset eval buffers after printing
        self._eval_seen.reset()
        self._eval_unseen.reset()

    # ── Training window summary ───────────────────────────────

    def print_episode_summary(self, episode):
        """Print rolling training window stats (called automatically every log_interval)."""
        stats = self._train_buf.summary()
        n     = len(self._train_buf)
        overall_win_rate = (
            self._total_wins / self._total_episodes * 100
            if self._total_episodes > 0 else 0.0
        )

        print(
            f"  [{_timestamp()}]  {self.algo_name}  |  "
            f"Ep {episode:>6d}  |  "
            f"Last {n} eps  |  "
            f"Return {stats['episode_return'][0]:+.2f} ± {stats['episode_return'][1]:.2f}  |  "
            f"Win% {overall_win_rate:.1f}"
            if "episode_return" in stats and "win" in stats
            else
            f"  [{_timestamp()}]  {self.algo_name}  |  "
            f"Ep {episode:>6d}  |  "
            f"Last {n} eps  |  "
            f"Return {stats.get('episode_return', (0,0))[0]:+.2f} ± {stats.get('episode_return', (0,0))[1]:.2f}"
        )

    # ── Final run summary ─────────────────────────────────────

    def print_final_summary(self):
        """Print overall stats at end of full training run."""
        _divider("═")
        print(f"  [{_timestamp()}]  FINAL SUMMARY  —  {self.algo_name}  (seed {self.seed})")
        _divider("═")
        print(f"  Total Episodes : {self._total_episodes}")
        if self._total_episodes > 0:
            print(f"  Overall Win Rate : {self._total_wins / self._total_episodes * 100:.1f}%")
        _divider("═")


# ──────────────────────────────────────────────────────────────
# Convenience function: compare two algos side by side
# (for the reproducibility table in your report)
# ──────────────────────────────────────────────────────────────

def print_comparison_table(results):
    """
    Print a side-by-side comparison table for the report.

    Parameters
    ----------
    results : dict
        {
            "OMIS":    {"return": (mean, std), "win_rate": (mean, std), "opp_acc": (mean, std)},
            "MAPPO":   {"return": (mean, std), "win_rate": (mean, std)},
            "MARLeOM": {"return": (mean, std), "win_rate": (mean, std), "opp_acc": (mean, std)},
        }

    Example
    -------
        print_comparison_table({
            "OMIS (paper)":  {"return": (155.2, 8.1), "win_rate": (0.72, 0.04)},
            "OMIS (ours)":   {"return": (148.7, 9.3), "win_rate": (0.69, 0.05)},
            "MAPPO":         {"return": (130.1, 11.2), "win_rate": (0.61, 0.06)},
        })
    """
    all_metrics = set()
    for v in results.values():
        all_metrics.update(v.keys())
    all_metrics = sorted(all_metrics)

    col_w = 22
    _divider("═")
    print("  ALGORITHM COMPARISON TABLE")
    _divider("═")

    header = f"  {'Algorithm':<20}" + "".join(f"  {m:<{col_w}}" for m in all_metrics)
    print(header)
    _divider()

    for algo, metrics in results.items():
        row = f"  {algo:<20}"
        for m in all_metrics:
            if m in metrics:
                mean, std = metrics[m]
                cell = f"{mean:.3f} ± {std:.3f}"
            else:
                cell = "—"
            row += f"  {cell:<{col_w}}"
        print(row)

    _divider("═")


# ──────────────────────────────────────────────────────────────
# Quick smoke test  (python log.py)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    logger = MARLLogger(algo_name="OMIS", log_interval=10, seed=42)

    # Simulate 30 training episodes
    for ep in range(1, 31):
        logger.log_episode(
            episode        = ep,
            episode_return = random.gauss(-80, 20),
            win            = random.randint(0, 1),
            opponent_type  = random.choice(["seen", "unseen"]),
        )

    # Simulate eval
    for _ in range(10):
        logger.log_eval(
            episode        = 30,
            episode_return = random.gauss(-75, 15),
            win            = random.randint(0, 1),
            opponent_acc   = random.uniform(0.45, 0.70),
            value_mse      = random.uniform(0.01, 0.08),
            opponent_type  = "seen",
        )
    for _ in range(10):
        logger.log_eval(
            episode        = 30,
            episode_return = random.gauss(-100, 20),
            win            = random.randint(0, 1),
            opponent_acc   = random.uniform(0.40, 0.60),
            value_mse      = random.uniform(0.02, 0.10),
            opponent_type  = "unseen",
        )

    logger.print_eval_summary(episode=30)
    logger.print_final_summary()

    # Comparison table demo
    print_comparison_table({
        "OMIS (paper)": {"return": (155.2, 8.1), "win_rate": (0.72, 0.04), "opp_acc": (0.60, 0.01)},
        "OMIS (ours)":  {"return": (148.7, 9.3), "win_rate": (0.69, 0.05), "opp_acc": (0.58, 0.02)},
        "MAPPO":        {"return": (130.1, 11.2), "win_rate": (0.61, 0.06)},
    })