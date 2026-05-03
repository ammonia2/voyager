from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc
    return plt


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    out: list[float] = []
    running = 0.0
    queue: list[float] = []
    for v in values:
        queue.append(v)
        running += v
        if len(queue) > window:
            running -= queue.pop(0)
        out.append(running / len(queue))
    return out


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _first_float(record: dict, keys: Iterable[str], default: float = 0.0) -> float:
    for key in keys:
        if key in record and record[key] is not None:
            return _safe_float(record[key], default=default)
    return default


def _save_line_plot(
    plt,
    x: Iterable[float],
    y: Iterable[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    label: str | None = None,
):
    x_vals = list(x)
    y_vals = list(y)
    if not x_vals or not y_vals:
        return

    plt.figure(figsize=(10, 5))
    if label:
        plt.plot(x_vals, y_vals, label=label)
        plt.legend()
    else:
        plt.plot(x_vals, y_vals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_two_line_plot(
    plt,
    x1: Iterable[float],
    y1: Iterable[float],
    x2: Iterable[float],
    y2: Iterable[float],
    title: str,
    xlabel: str,
    ylabel: str,
    label1: str,
    label2: str,
    out_path: Path,
):
    x1_vals, y1_vals = list(x1), list(y1)
    x2_vals, y2_vals = list(x2), list(y2)
    if not x1_vals or not y1_vals or not x2_vals or not y2_vals:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(x1_vals, y1_vals, label=label1)
    plt.plot(x2_vals, y2_vals, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_hist(
    plt,
    values: list[float],
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 30,
):
    if not values:
        return
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins, alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _extract(records: list[dict], record_type: str) -> list[dict]:
    return [r for r in records if r.get("record_type") == record_type]


def _record_index(record: dict) -> int:
    value = record.get("update", record.get("step", record.get("episode", 0)))
    return int(value)


def _plot_optional_series(
    plt,
    out_dir: Path,
    x: list[int],
    y: list[float],
    title: str,
    ylabel: str,
    filename: str,
):
    if x and y and any(v != 0.0 for v in y):
        _save_line_plot(plt, x, y, title, "Update", ylabel, out_dir / filename)


def main():
    parser = argparse.ArgumentParser(description="Plot MASAC predator/prey JSONL logs")
    parser.add_argument(
        "--predator-log",
        default="checkpoints/masac_predator_metrics.jsonl",
        help="Path to predator JSONL log",
    )
    parser.add_argument(
        "--prey-log",
        default="checkpoints/masac_prey_metrics.jsonl",
        help="Path to prey JSONL log",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/masac",
        help="Output folder for graphs (relative to repo root by default)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        help="Window size for rolling mean plots",
    )
    args = parser.parse_args()

    plt = _require_matplotlib()

    predator_log = Path(args.predator_log)
    prey_log = Path(args.prey_log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not predator_log.exists():
        raise SystemExit(f"Predator log not found: {predator_log}")
    if not prey_log.exists():
        raise SystemExit(f"Prey log not found: {prey_log}")

    pred_records = _load_jsonl(predator_log)
    prey_records = _load_jsonl(prey_log)

    pred_ep = _extract(pred_records, "episode")
    pred_step = _extract(pred_records, "step")
    prey_ep = _extract(prey_records, "episode")
    prey_step = _extract(prey_records, "step")

    # Episode series (kept MAPPO-like for consistency across experiments).
    pred_episode = [int(r["episode"]) for r in pred_ep]
    pred_return = [_first_float(r, ["episode_return", "return"]) for r in pred_ep]
    pred_win = [_first_float(r, ["win", "tagged", "is_win"]) for r in pred_ep]

    prey_episode = [int(r["episode"]) for r in prey_ep]
    prey_return = [_first_float(r, ["episode_return", "return"]) for r in prey_ep]
    prey_win = [_first_float(r, ["win", "escaped", "is_win"]) for r in prey_ep]
    prey_escape = [_first_float(r, ["escape_rate", "win", "escaped"]) for r in prey_ep]
    prey_survival = [_first_float(r, ["survival_time", "steps_survived", "episode_len"]) for r in prey_ep]
    prey_policy_entropy_ep = [_first_float(r, ["policy_entropy", "entropy", "policyEntropy"]) for r in prey_ep]

    # Basic returns
    _save_line_plot(
        plt,
        pred_episode,
        pred_return,
        "Predator Episode Return",
        "Episode",
        "Return",
        out_dir / "predator_return.png",
    )
    _save_line_plot(
        plt,
        prey_episode,
        prey_return,
        "Prey Episode Return",
        "Episode",
        "Return",
        out_dir / "prey_return.png",
    )
    _save_two_line_plot(
        plt,
        pred_episode,
        pred_return,
        prey_episode,
        prey_return,
        "Episode Return Comparison",
        "Episode",
        "Return",
        "Predator",
        "Prey",
        out_dir / "return_comparison.png",
    )

    # Rolling return
    _save_two_line_plot(
        plt,
        pred_episode,
        _rolling_mean(pred_return, args.rolling_window),
        prey_episode,
        _rolling_mean(prey_return, args.rolling_window),
        f"Rolling Return (window={args.rolling_window})",
        "Episode",
        "Return",
        "Predator",
        "Prey",
        out_dir / "rolling_return_comparison.png",
    )

    # Win / escape
    if pred_win:
        pred_cum_win = _rolling_mean(pred_win, len(pred_win))
        _save_line_plot(
            plt,
            pred_episode,
            pred_cum_win,
            "Predator Cumulative Win Rate",
            "Episode",
            "Win Rate",
            out_dir / "predator_cumulative_win_rate.png",
        )
        _save_line_plot(
            plt,
            pred_episode,
            _rolling_mean(pred_win, args.rolling_window),
            f"Predator Rolling Win Rate (window={args.rolling_window})",
            "Episode",
            "Win Rate",
            out_dir / "predator_rolling_win_rate.png",
        )

    if prey_win:
        prey_cum_win = _rolling_mean(prey_win, len(prey_win))
        _save_line_plot(
            plt,
            prey_episode,
            prey_cum_win,
            "Prey Cumulative Escape Rate",
            "Episode",
            "Escape Rate",
            out_dir / "prey_cumulative_escape_rate.png",
        )
    _save_line_plot(
        plt,
        prey_episode,
        _rolling_mean(prey_escape, args.rolling_window),
        f"Prey Rolling Escape Rate (window={args.rolling_window})",
        "Episode",
        "Escape Rate",
        out_dir / "prey_rolling_escape_rate.png",
    )

    # Prey episode metrics
    _save_line_plot(
        plt,
        prey_episode,
        prey_survival,
        "Prey Survival Time",
        "Episode",
        "Steps",
        out_dir / "prey_survival_time.png",
    )
    if prey_policy_entropy_ep and any(x != 0.0 for x in prey_policy_entropy_ep):
        _save_line_plot(
            plt,
            prey_episode,
            prey_policy_entropy_ep,
            "Prey Policy Entropy (Episode)",
            "Episode",
            "Entropy",
            out_dir / "prey_policy_entropy_episode.png",
        )

    # Histograms
    _save_hist(
        plt,
        pred_return,
        "Predator Return Distribution",
        "Episode Return",
        out_dir / "predator_return_hist.png",
    )
    _save_hist(
        plt,
        prey_return,
        "Prey Return Distribution",
        "Episode Return",
        out_dir / "prey_return_hist.png",
    )
    _save_hist(
        plt,
        prey_survival,
        "Prey Survival Time Distribution",
        "Survival Time",
        out_dir / "prey_survival_hist.png",
    )

    # Step-level predator metrics (MASAC aliases + MAPPO aliases for compatibility).
    if pred_step:
        pred_update = [_record_index(r) for r in pred_step]
        predator_actor_loss = [
            _first_float(r, ["actorLoss", "policyLoss", "actor_loss", "loss_actor", "loss"])
            for r in pred_step
        ]
        predator_critic_loss = [
            _first_float(r, ["criticLoss", "valueLoss", "critic_loss", "loss_critic", "qLoss"])
            for r in pred_step
        ]
        predator_entropy = [
            _first_float(r, ["entropy", "policyEntropy", "policy_entropy"]) for r in pred_step
        ]
        predator_om_loss = [
            _first_float(r, ["omLoss", "om0Loss", "om0", "opponentLoss", "opponent_loss"]) for r in pred_step
        ]
        predator_alpha = [_first_float(r, ["alpha", "temperature"]) for r in pred_step]
        predator_alpha_loss = [_first_float(r, ["alphaLoss", "temperatureLoss"]) for r in pred_step]
        predator_sec_per_update = [
            _first_float(r, ["sec_per_up", "sec_per_ep", "seconds_per_update", "seconds_per_episode"])
            for r in pred_step
        ]

        _plot_optional_series(
            plt,
            out_dir,
            pred_update,
            predator_actor_loss,
            "Predator Actor/Policy Loss",
            "Loss",
            "predator_policy_loss.png",
        )
        _plot_optional_series(
            plt,
            out_dir,
            pred_update,
            predator_critic_loss,
            "Predator Critic/Value Loss",
            "Loss",
            "predator_value_loss.png",
        )
        _plot_optional_series(
            plt,
            out_dir,
            pred_update,
            predator_entropy,
            "Predator Entropy",
            "Entropy",
            "predator_entropy_step.png",
        )
        _plot_optional_series(
            plt,
            out_dir,
            pred_update,
            predator_om_loss,
            "Predator OM Loss",
            "Loss",
            "predator_om_loss.png",
        )
        _plot_optional_series(
            plt,
            out_dir,
            pred_update,
            predator_alpha,
            "Predator Temperature (Alpha)",
            "Alpha",
            "predator_alpha.png",
        )
        _plot_optional_series(
            plt,
            out_dir,
            pred_update,
            predator_alpha_loss,
            "Predator Alpha Loss",
            "Loss",
            "predator_alpha_loss.png",
        )
        _plot_optional_series(
            plt,
            out_dir,
            pred_update,
            predator_sec_per_update,
            "Predator Seconds per Update",
            "Seconds",
            "predator_sec_per_update.png",
        )

    # Step-level prey metrics
    if prey_step:
        prey_update = [_record_index(r) for r in prey_step]
        prey_policy_entropy_step = [
            _first_float(r, ["policyEntropy", "policy_entropy", "entropy"]) for r in prey_step
        ]
        prey_adv_mse = [
            _first_float(r, ["advantageMseSanity", "adv_mse", "value_mse"]) for r in prey_step
        ]
        prey_sec_per_update = [
            _first_float(r, ["sec_per_up", "sec_per_ep", "seconds_per_update", "seconds_per_episode"])
            for r in prey_step
        ]

        _plot_optional_series(
            plt,
            out_dir,
            prey_update,
            prey_policy_entropy_step,
            "Prey Policy Entropy (Step)",
            "Entropy",
            "prey_policy_entropy_step.png",
        )
        _plot_optional_series(
            plt,
            out_dir,
            prey_update,
            prey_adv_mse,
            "Prey Advantage MSE Sanity",
            "MSE",
            "prey_advantage_mse.png",
        )
        _plot_optional_series(
            plt,
            out_dir,
            prey_update,
            prey_sec_per_update,
            "Prey Seconds per Update",
            "Seconds",
            "prey_sec_per_update.png",
        )

    print(f"Saved MASAC plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
