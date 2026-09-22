#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from compare_mux_methods import DISPLAY_NAMES, METHODS


COLORS = {
    "channel-concat": "#2468A2",
    "spatial-tile": "#2E8B57",
    "orthogonal-code": "#8A2D5D",
    "learned-feature-code": "#C18A00",
}
MARKERS = {
    "channel-concat": "o",
    "spatial-tile": "s",
    "orthogonal-code": "^",
    "learned-feature-code": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signals", type=int, nargs="+", default=[2, 3, 5, 10, 15, 20]
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--images-per-batch", type=int, default=768)
    parser.add_argument("--training-size", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("mux_method_signal_sweep")
    )
    parser.add_argument(
        "--three-signal-dir",
        type=Path,
        default=Path("mux_method_comparison_improved_10ep"),
        help="Existing 10-epoch result directory that can be reused for N=3.",
    )
    parser.add_argument("--reuse-three-signal", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def result_directory(args: argparse.Namespace, signals: int) -> Path:
    if args.reuse_three_signal and signals == 3 and args.three_signal_dir.exists():
        return args.three_signal_dir
    return args.output_dir / f"signals_{signals}"


def train_signal_count(args: argparse.Namespace, signals: int) -> list[dict]:
    run_dir = result_directory(args, signals)
    result_path = run_dir / "mux_method_results.json"
    if not (args.reuse_existing and result_path.exists()):
        run_dir.mkdir(parents=True, exist_ok=True)
        batch_size = max(1, args.images_per_batch // signals)
        command = [
            sys.executable,
            "compare_mux_methods.py",
            "--slots",
            str(signals),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(batch_size),
            "--workers",
            str(args.workers),
            "--seed",
            str(args.seed),
            "--data-dir",
            str(args.data_dir),
            "--training-size",
            str(args.training_size),
            "--output-dir",
            str(run_dir),
        ]
        if args.reuse_existing:
            command.append("--reuse-existing")
        print(f"\nTraining four methods with {signals} signals", flush=True)
        subprocess.run(command, check=True)

    with result_path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    for row in rows:
        row["signal_count"] = signals
    return rows


def save_results(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "method_signal_sweep_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(rows, handle, indent=2)
    fields = [
        "signal_count",
        "method",
        "display_name",
        "mean_accuracy",
        "all_correct_accuracy",
        "parameters",
        "training_seconds",
        "images_per_second",
        "best_epoch",
    ]
    with (output_dir / "method_signal_sweep_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(rows: list[dict], output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(8.2, 5.1))
    for method in METHODS:
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["signal_count"],
        )
        axis.plot(
            [row["signal_count"] for row in method_rows],
            [100 * row["mean_accuracy"] for row in method_rows],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.3,
            markersize=6,
            label=DISPLAY_NAMES[method],
        )

    signals = sorted({row["signal_count"] for row in rows})
    axis.set_xlabel("Number of Multiplexed Signals")
    axis.set_ylabel("Test Accuracy (%)")
    axis.set_title("Multiplexing Method Accuracy by Signal Count")
    axis.set_xticks(signals)
    axis.set_ylim(0, 101)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    axis.legend(loc="upper right", frameon=True)
    figure.tight_layout()
    output = output_dir / "four_methods_accuracy_vs_signals.png"
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    args = parse_args()
    if any(signals < 2 for signals in args.signals):
        raise ValueError("Every method comparison requires at least two signals")
    all_rows = []
    for signals in args.signals:
        all_rows.extend(train_signal_count(args, signals))
    all_rows.sort(key=lambda row: (row["signal_count"], row["method"]))
    save_results(all_rows, args.output_dir)
    figure_path = draw_figure(all_rows, args.output_dir)
    print(f"\nFigure: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
