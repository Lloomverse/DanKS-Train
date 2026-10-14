#!/usr/bin/env python3
"""Render open-source README evidence charts from exported online metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-steps-per-sec",
        type=float,
        required=True,
        help="Naive / previous online pipeline throughput for the speedup bar",
    )
    parser.add_argument("--baseline-label", type=str, required=True)
    parser.add_argument("--current-label", type=str, required=True)
    return parser.parse_args()


def _load_metrics(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, float]] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        required = {
            "update",
            "steps_per_sec",
            "loss",
            "entropy",
            "ef",
        }
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise RuntimeError(
                f"metrics csv missing columns: {sorted(required - set(reader.fieldnames or []))}"
            )
        for raw in reader:
            rows.append(
                {
                    "update": float(raw["update"]),
                    "steps_per_sec": float(raw["steps_per_sec"]),
                    "loss": float(raw["loss"]),
                    "entropy": float(raw["entropy"]),
                    "ef": float(raw["ef"]),
                }
            )
    if not rows:
        raise RuntimeError(f"metrics csv is empty: {path}")
    return rows


def _rolling_median(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if window > len(values):
        raise ValueError(f"window {window} > series length {len(values)}")
    out: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = sorted(values[start : index + 1])
        mid = len(chunk) // 2
        if len(chunk) % 2 == 1:
            out.append(chunk[mid])
        else:
            out.append(0.5 * (chunk[mid - 1] + chunk[mid]))
    return out


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 11,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.8,
        }
    )


def plot_speedup(
    *,
    baseline: float,
    current: float,
    baseline_label: str,
    current_label: str,
    out_path: Path,
) -> None:
    if baseline <= 0.0 or current <= 0.0:
        raise ValueError(f"throughputs must be > 0, got {baseline}, {current}")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [baseline_label, current_label]
    values = [baseline, current]
    colors = ["#8a8a8a", "#c23b22"]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_ylabel("Training throughput (env step / s)")
    ax.set_title("Online self-play throughput")
    ratio = current / baseline
    ax.annotate(
        f"{ratio:.1f}×",
        xy=(1, current),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#c23b22",
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0.0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_throughput(rows: list[dict[str, float]], out_path: Path) -> None:
    updates = [row["update"] for row in rows]
    sps = [row["steps_per_sec"] for row in rows]
    window = min(101, max(5, len(sps) // 40))
    smooth = _rolling_median(sps, window)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(updates, sps, color="#bbbbbb", linewidth=0.6, label="per update")
    ax.plot(updates, smooth, color="#c23b22", linewidth=2.0, label=f"rolling median ({window})")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("env step / s")
    ax.set_title("Online training throughput over a long run")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_convergence(rows: list[dict[str, float]], out_path: Path) -> None:
    updates = [row["update"] for row in rows]
    loss = [row["loss"] for row in rows]
    entropy = [row["entropy"] for row in rows]
    ef = [row["ef"] for row in rows]
    window = min(101, max(5, len(loss) // 40))
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.2), sharex=True)
    series = (
        (axes[0], loss, "PPO loss", "#222222"),
        (axes[1], entropy, "policy entropy", "#2a6fbb"),
        (axes[2], ef, "explained variance", "#2f9e44"),
    )
    for ax, values, title, color in series:
        ax.plot(updates, values, color="#cccccc", linewidth=0.5)
        ax.plot(updates, _rolling_median(values, window), color=color, linewidth=2.0)
        ax.set_ylabel(title)
    axes[-1].set_xlabel("PPO update")
    axes[0].set_title("Online training dynamics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = _load_metrics(args.metrics_csv)
    warm = rows[max(1, len(rows) // 20) :]
    current = sorted(row["steps_per_sec"] for row in warm)[len(warm) // 2]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    plot_speedup(
        baseline=args.baseline_steps_per_sec,
        current=current,
        baseline_label=args.baseline_label,
        current_label=args.current_label,
        out_path=args.out_dir / "online-training-speedup.png",
    )
    plot_throughput(rows, args.out_dir / "online-training-throughput.png")
    plot_convergence(rows, args.out_dir / "online-training-convergence.png")
    summary = args.out_dir / "evidence" / "speed_comparison.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "steps_per_sec", "notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "label": args.baseline_label,
                "steps_per_sec": f"{args.baseline_steps_per_sec:.2f}",
                "notes": "naive multi-table online batch (152 tables)",
            }
        )
        writer.writerow(
            {
                "label": args.current_label,
                "steps_per_sec": f"{current:.2f}",
                "notes": (
                    f"median over warm updates "
                    f"{int(warm[0]['update'])}..{int(warm[-1]['update'])}"
                ),
            }
        )
    print(f"baseline={args.baseline_steps_per_sec:.1f} current_median={current:.1f}", flush=True)
    print(f"wrote charts under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
