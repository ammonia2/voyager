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
    plt.figure(figsize=(10, 5))
    if label:
        plt.plot(list(x), list(y), label=label)
        plt.legend()
    else:
        plt.plot(list(x), list(y))
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
    plt.figure(figsize=(10, 5))
    plt.plot(list(x1), list(y1), label=label1)
    plt.plot(list(x2), list(y2), label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_hist(plt, values: list[float], title: str, xlabel: str, out_path: Path, bins: int = 30):
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


def main():
    parser = argparse.ArgumentParser(description="Plot OMIS predator/prey JSONL logs")
    parser.add_argument(
        "--predator-log",
        default="checkpoints/omis_predator_metrics.jsonl",
        help="Path to predator JSONL log",
    )
    parser.add_argument(
        "--prey-log",
        default="checkpoints/omis_prey_metrics.jsonl",
        help="Path to prey JSONL log",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/omis",
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

    # Episode series
    pred_episode = [int(r["episode"]) for r in pred_ep]
    pred_return = [_safe_float(r.get("episode_return")) for r in pred_ep]
    pred_win = [_safe_float(r.get("win")) for r in pred_ep]

    prey_episode = [int(r["episode"]) for r in prey_ep]
    prey_return = [_safe_float(r.get("episode_return")) for r in prey_ep]
    prey_win = [_safe_float(r.get("win")) for r in prey_ep]
    prey_escape = [_safe_float(r.get("escape_rate")) for r in prey_ep]
    prey_survival = [_safe_float(r.get("survival_time")) for r in prey_ep]
    prey_opponent_acc = [_safe_float(r.get("opponent_acc")) for r in prey_ep]

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
    pred_cum_win = _rolling_mean(pred_win, len(pred_win))
    prey_cum_win = _rolling_mean(prey_win, len(prey_win))
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
        prey_episode,
        prey_cum_win,
        "Prey Cumulative Escape Rate",
        "Episode",
        "Escape Rate",
        out_dir / "prey_cumulative_escape_rate.png",
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
    if prey_opponent_acc and any(x > 0 for x in prey_opponent_acc):
        _save_line_plot(
            plt,
            prey_episode,
            prey_opponent_acc,
            "Prey Opponent Action Accuracy",
            "Episode",
            "Accuracy",
            out_dir / "prey_opponent_accuracy.png",
        )

    # Histograms
    _save_hist(plt, pred_return, "Predator Return Distribution", "Episode Return", out_dir / "predator_return_hist.png")
    _save_hist(plt, prey_return, "Prey Return Distribution", "Episode Return", out_dir / "prey_return_hist.png")
    _save_hist(plt, prey_survival, "Prey Survival Time Distribution", "Survival Time", out_dir / "prey_survival_hist.png")

    # Step-level predator metrics
    if pred_step:
        pred_update = [int(r["update"]) for r in pred_step]
        _save_line_plot(plt, pred_update, [_safe_float(r.get("loss")) for r in pred_step],
                        "Predator Loss", "Update", "Loss", out_dir / "predator_loss.png")

    # Step-level prey metrics
    if prey_step:
        prey_update = [int(r["update"]) for r in prey_step]
        if any(_safe_float(r.get("sec_per_up")) > 0 for r in prey_step):
            _save_line_plot(plt, prey_update, [_safe_float(r.get("sec_per_up")) for r in prey_step],
                            "Prey Seconds per Update", "Update", "Seconds", out_dir / "prey_sec_per_update.png")

    print(f"Saved OMIS plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
