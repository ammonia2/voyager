from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence


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
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        out.append(running / len(queue))
    return out


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_float(record: dict, keys: Iterable[str], default: float = 0.0) -> float:
    for key in keys:
        if key in record and record[key] is not None:
            return _safe_float(record[key], default=default)
    return default


def _extract(records: list[dict], record_type: str) -> list[dict]:
    return [record for record in records if record.get("record_type") == record_type]


def _record_index(record: dict) -> int:
    value = record.get("update", record.get("step", record.get("episode", 0)))
    return int(value)


def _segment_label(segment_index: int) -> str:
    return f"Prey {segment_index + 1}"


def _group_by_segment(records: Sequence[dict], segment_size: int) -> list[list[dict]]:
    grouped: list[list[dict]] = []
    for record in records:
        episode = int(record.get("episode", 0))
        if episode <= 0:
            continue
        segment_index = (episode - 1) // segment_size
        while len(grouped) <= segment_index:
            grouped.append([])
        grouped[segment_index].append(record)
    return grouped


def _segment_bounds(total_episodes: int, segment_size: int) -> list[int]:
    bounds = []
    boundary = segment_size
    while boundary < total_episodes:
        bounds.append(boundary)
        boundary += segment_size
    return bounds


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
    x_values = list(x)
    y_values = list(y)
    if not x_values or not y_values:
        return

    plt.figure(figsize=(10, 5))
    if label:
        plt.plot(x_values, y_values, label=label)
        plt.legend()
    else:
        plt.plot(x_values, y_values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_hist(plt, values: list[float], title: str, xlabel: str, out_path: Path, bins: int = 30):
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


def _style_plot(ax, title: str, xlabel: str, ylabel: str, bounds: list[int] | None = None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if bounds:
        for boundary in bounds:
            ax.axvline(boundary + 0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.35)


def _save_segmented_single_plot(
    plt,
    grouped_records: list[list[dict]],
    value_keys: Sequence[str],
    title: str,
    ylabel: str,
    out_path: Path,
    segment_size: int,
    rolling_window: int | None = None,
    cumulative: bool = False,
):
    if not grouped_records:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    bounds = _segment_bounds(sum(len(group) for group in grouped_records), segment_size)

    for segment_index, segment_records in enumerate(grouped_records):
        if not segment_records:
            continue
        episodes = [int(record["episode"]) for record in segment_records]
        values = [_first_float(record, value_keys) for record in segment_records]
        if rolling_window is not None:
            values = _rolling_mean(values, rolling_window)
        elif cumulative:
            values = _rolling_mean(values, len(values))

        ax.plot(
            episodes,
            values,
            color=cmap(segment_index % 10),
            label=_segment_label(segment_index),
            linewidth=2.0,
        )

    _style_plot(ax, title, "Episode", ylabel, bounds=bounds)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_segmented_dual_plot(
    plt,
    predator_groups: list[list[dict]],
    prey_groups: list[list[dict]],
    predator_keys: Sequence[str],
    prey_keys: Sequence[str],
    title: str,
    ylabel: str,
    out_path: Path,
    segment_size: int,
    rolling_window: int | None = None,
):
    if not predator_groups or not prey_groups:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    total_episodes = max(
        sum(len(group) for group in predator_groups),
        sum(len(group) for group in prey_groups),
    )
    bounds = _segment_bounds(total_episodes, segment_size)

    for segment_index, (pred_group, prey_group) in enumerate(zip(predator_groups, prey_groups)):
        if not pred_group or not prey_group:
            continue

        color = cmap(segment_index % 10)
        pred_x = [int(record["episode"]) for record in pred_group]
        prey_x = [int(record["episode"]) for record in prey_group]
        pred_y = [_first_float(record, predator_keys) for record in pred_group]
        prey_y = [_first_float(record, prey_keys) for record in prey_group]

        if rolling_window is not None:
            pred_y = _rolling_mean(pred_y, rolling_window)
            prey_y = _rolling_mean(prey_y, rolling_window)

        ax.plot(pred_x, pred_y, color=color, linestyle="-", linewidth=2.0, label=f"{_segment_label(segment_index)} Predator")
        ax.plot(prey_x, prey_y, color=color, linestyle="--", linewidth=2.0, label=f"{_segment_label(segment_index)} Prey")

    _style_plot(ax, title, "Episode", ylabel, bounds=bounds)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_optional_step_series(
    plt,
    step_records: list[dict],
    value_keys: Sequence[str],
    title: str,
    ylabel: str,
    out_path: Path,
):
    if not step_records:
        return
    step_x = [_record_index(record) for record in step_records]
    step_y = [_first_float(record, value_keys) for record in step_records]
    if not any(value != 0.0 for value in step_y):
        return
    _save_line_plot(plt, step_x, step_y, title, "Update", ylabel, out_path)


def main():
    parser = argparse.ArgumentParser(description="Plot OMIS unseen predator/prey JSONL logs with prey swaps")
    parser.add_argument(
        "--predator-log",
        default="checkpoints/omis_predator_metrics_unseen.jsonl",
        help="Path to unseen predator JSONL log",
    )
    parser.add_argument(
        "--prey-log",
        default="checkpoints/omis_prey_metrics_unseen.jsonl",
        help="Path to unseen prey JSONL log",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/omis_unseen",
        help="Output folder for graphs (relative to repo root by default)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Window size for rolling mean plots",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=30,
        help="Episodes per unseen prey before the evaluation swaps to the next prey",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=90,
        help="Maximum episode count to plot across the concatenated unseen prey run",
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

    pred_ep = [record for record in pred_ep if int(record.get("episode", 0)) <= args.max_episodes]
    prey_ep = [record for record in prey_ep if int(record.get("episode", 0)) <= args.max_episodes]
    pred_step = [record for record in pred_step if _record_index(record) <= args.max_episodes]
    prey_step = [record for record in prey_step if _record_index(record) <= args.max_episodes]

    pred_groups = _group_by_segment(pred_ep, args.segment_size)
    prey_groups = _group_by_segment(prey_ep, args.segment_size)

    pred_return = [_safe_float(record.get("episode_return")) for record in pred_ep]
    prey_return = [_safe_float(record.get("episode_return")) for record in prey_ep]
    prey_survival = [_safe_float(record.get("survival_time")) for record in prey_ep]

    _save_segmented_dual_plot(
        plt,
        pred_groups,
        prey_groups,
        ["episode_return", "return"],
        ["episode_return", "return"],
        "OMIS Unseen Episode Return Comparison",
        "Return",
        out_dir / "return_comparison.png",
        args.segment_size,
    )
    _save_segmented_dual_plot(
        plt,
        pred_groups,
        prey_groups,
        ["episode_return", "return"],
        ["episode_return", "return"],
        f"OMIS Unseen Rolling Return (window={args.rolling_window})",
        "Return",
        out_dir / "rolling_return_comparison.png",
        args.segment_size,
        rolling_window=args.rolling_window,
    )

    _save_segmented_single_plot(
        plt,
        pred_groups,
        ["win", "tagged", "is_win"],
        "OMIS Unseen Predator Win Rate by Prey",
        "Win Rate",
        out_dir / "predator_cumulative_win_rate.png",
        args.segment_size,
        cumulative=True,
    )
    _save_segmented_single_plot(
        plt,
        prey_groups,
        ["win", "escaped", "is_win"],
        "OMIS Unseen Prey Escape Rate by Prey",
        "Escape Rate",
        out_dir / "prey_cumulative_escape_rate.png",
        args.segment_size,
        cumulative=True,
    )
    _save_segmented_single_plot(
        plt,
        pred_groups,
        ["win", "tagged", "is_win"],
        f"OMIS Unseen Predator Rolling Win Rate (window={args.rolling_window})",
        "Win Rate",
        out_dir / "predator_rolling_win_rate.png",
        args.segment_size,
        rolling_window=args.rolling_window,
    )
    _save_segmented_single_plot(
        plt,
        prey_groups,
        ["win", "escaped", "is_win"],
        f"OMIS Unseen Prey Rolling Escape Rate (window={args.rolling_window})",
        "Escape Rate",
        out_dir / "prey_rolling_escape_rate.png",
        args.segment_size,
        rolling_window=args.rolling_window,
    )
    _save_segmented_single_plot(
        plt,
        prey_groups,
        ["survival_time", "steps_survived", "episode_len"],
        "OMIS Unseen Prey Survival Time by Prey",
        "Steps",
        out_dir / "prey_survival_time.png",
        args.segment_size,
    )

    if any(record.get("opponent_acc") is not None for record in prey_ep):
        _save_segmented_single_plot(
            plt,
            prey_groups,
            ["opponent_acc"],
            "OMIS Unseen Prey Opponent Action Accuracy by Prey",
            "Accuracy",
            out_dir / "prey_opponent_accuracy.png",
            args.segment_size,
        )

    _save_hist(plt, pred_return, "OMIS Unseen Predator Return Distribution", "Episode Return", out_dir / "predator_return_hist.png")
    _save_hist(plt, prey_return, "OMIS Unseen Prey Return Distribution", "Episode Return", out_dir / "prey_return_hist.png")
    _save_hist(plt, prey_survival, "OMIS Unseen Prey Survival Time Distribution", "Survival Time", out_dir / "prey_survival_hist.png")

    if pred_step:
        _plot_optional_step_series(
            plt,
            pred_step,
            ["loss", "criticLoss", "valueLoss"],
            "OMIS Unseen Predator Loss",
            "Loss",
            out_dir / "predator_loss.png",
        )

    if prey_step:
        _plot_optional_step_series(
            plt,
            prey_step,
            ["sec_per_up", "sec_per_ep", "seconds_per_update", "seconds_per_episode"],
            "OMIS Unseen Prey Seconds per Update",
            "Seconds",
            out_dir / "prey_sec_per_update.png",
        )

    print(f"Saved OMIS unseen plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()