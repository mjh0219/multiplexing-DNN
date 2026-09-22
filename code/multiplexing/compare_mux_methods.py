#!/usr/bin/env python3
"""Compare several MUX/DEMUX strategies on the official MNIST test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from mux_image_experiment import choose_device, split_datasets


METHODS = (
    "channel-concat",
    "spatial-tile",
    "orthogonal-code",
    "learned-feature-code",
)

DISPLAY_NAMES = {
    "channel-concat": "Channel concatenation",
    "spatial-tile": "Spatial tiling",
    "orthogonal-code": "Orthogonal-code superposition",
    "learned-feature-code": "Learned feature code",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--training-size",
        type=int,
        default=0,
        help="Use this many post-validation training images; 0 uses all available images.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("mux_method_comparison"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FlexibleCarrierCNN(nn.Module):
    """Shared classifier for carriers with different channel/spatial layouts."""

    def __init__(self, in_channels: int, output_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(self, carrier: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(carrier))


class FixedCarrierMux(nn.Module):
    """Use a fixed input MUX and reshape output logits as the DEMUX."""

    def __init__(self, method: str, slots: int, classes: int, seed: int) -> None:
        super().__init__()
        self.method = method
        self.slots = slots
        self.classes = classes
        code_width = 1 << (slots - 1).bit_length()
        if method == "channel-concat":
            in_channels = slots
        elif method == "orthogonal-code":
            in_channels = code_width
        else:
            in_channels = 1
        self.network = FlexibleCarrierCNN(in_channels, slots * classes)
        hadamard = torch.ones(1, 1)
        while hadamard.shape[0] < code_width:
            hadamard = torch.cat(
                [
                    torch.cat([hadamard, hadamard], dim=1),
                    torch.cat([hadamard, -hadamard], dim=1),
                ],
                dim=0,
            )
        self.register_buffer("orthogonal_codes", hadamard[:slots])

    def mux(self, images: torch.Tensor) -> torch.Tensor:
        if self.method == "channel-concat":
            batch, slots, channels, height, width = images.shape
            return images.reshape(batch, slots * channels, height, width)
        if self.method == "spatial-tile":
            return torch.cat(images.unbind(dim=1), dim=-1)
        if self.method == "orthogonal-code":
            return torch.einsum(
                "bshw,sk->bkhw", images.squeeze(2), self.orthogonal_codes
            ) / math.sqrt(
                self.orthogonal_codes.shape[1]
            )
        raise ValueError(f"Unknown fixed MUX method: {self.method}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.network(self.mux(images))
        return logits.reshape(images.shape[0], self.slots, self.classes)


class LearnedFeatureCodeMux(nn.Module):
    """Bind per-image features to learned slot codes before superposition."""

    def __init__(self, slots: int, classes: int) -> None:
        super().__init__()
        self.slots = slots
        self.classes = classes
        self.slot_codes = nn.Parameter(torch.randn(slots, 32) * 0.2)
        self.input_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.shared_network = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.output_demux = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, slots * classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, slots, channels, height, width = images.shape
        features = self.input_encoder(images.reshape(batch * slots, channels, height, width))
        features = features.reshape(batch, slots, 32, features.shape[-2], features.shape[-1])
        codes = torch.tanh(self.slot_codes)[None, :, :, None, None]
        carrier = (features * codes).sum(dim=1) / math.sqrt(self.slots)
        logits = self.output_demux(self.shared_network(carrier))
        return logits.reshape(batch, self.slots, self.classes)


def build_model(method: str, slots: int, classes: int, seed: int) -> nn.Module:
    if method == "learned-feature-code":
        return LearnedFeatureCodeMux(slots, classes)
    return FixedCarrierMux(method, slots, classes, seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0
    all_correct = 0
    total_groups = 0
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels, _active in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            if training:
                loss.backward()
                optimizer.step()

            correct = logits.argmax(dim=-1).eq(labels)
            predictions = labels.numel()
            total_loss += float(loss.item()) * predictions
            total_correct += int(correct.sum().item())
            total_predictions += predictions
            all_correct += int(correct.all(dim=1).sum().item())
            total_groups += labels.shape[0]

    elapsed = time.perf_counter() - started
    return {
        "loss": total_loss / total_predictions,
        "mean_accuracy": total_correct / total_predictions,
        "all_correct_accuracy": all_correct / total_groups,
        "elapsed_seconds": elapsed,
        "images_per_second": total_predictions / elapsed,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_comparison_datasets(root: Path, download: bool):
    train_raw = datasets.MNIST(root=root, train=True, download=download)
    test_raw = datasets.MNIST(root=root, train=False, download=download)

    def normalized_tensor_dataset(dataset: datasets.MNIST) -> TensorDataset:
        images = dataset.data.unsqueeze(1).float().div_(255.0)
        images.sub_(0.1307).div_(0.3081)
        return TensorDataset(images, dataset.targets)

    train_data = normalized_tensor_dataset(train_raw)
    test_data = normalized_tensor_dataset(test_raw)
    return train_data, train_data, test_data


def train_method(
    method: str,
    args: argparse.Namespace,
    train_data,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    result_path = args.output_dir / f"result_{method}.json"
    if args.reuse_existing and result_path.exists():
        with result_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    set_seed(args.seed)
    model = build_model(method, args.slots, 10, args.seed).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_accuracy = -1.0
    best_epoch = 0
    training_seconds = 0.0
    checkpoint_path = args.output_dir / f"best_{method}.pt"

    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, device, None)
        scheduler.step()
        training_seconds += train_metrics["elapsed_seconds"]
        print(
            f"[{method}] epoch {epoch + 1:02d}/{args.epochs:02d} "
            f"train={train_metrics['mean_accuracy']:.4f} "
            f"validation={validation_metrics['mean_accuracy']:.4f}",
            flush=True,
        )
        if validation_metrics["mean_accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["mean_accuracy"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = run_epoch(model, test_loader, device, None)
    result = {
        "method": method,
        "display_name": DISPLAY_NAMES[method],
        "slots": args.slots,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "parameters": count_parameters(model),
        "training_seconds": training_seconds,
        **test_metrics,
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def save_results(results: list[dict], output_dir: Path) -> None:
    with (output_dir / "mux_method_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    fields = list(results[0])
    with (output_dir / "mux_method_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def draw_figure(results: list[dict], output_dir: Path) -> Path:
    names = [result["display_name"] for result in results]
    accuracy = [100 * result["mean_accuracy"] for result in results]
    all_correct = [100 * result["all_correct_accuracy"] for result in results]
    positions = np.arange(len(results))
    width = 0.36

    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    bars1 = axis.bar(
        positions - width / 2,
        accuracy,
        width,
        color="#2468A2",
        label="average signal accuracy",
    )
    bars2 = axis.bar(
        positions + width / 2,
        all_correct,
        width,
        color="#2E8B57",
        label="All three signals correct",
    )
    axis.set_ylabel("Test Accuracy (%)")
    axis.set_xlabel("MUX/DEMUX Method")
    axis.set_title("Accuracy Comparison of Multiplexing Methods")
    axis.set_xticks(positions, names, rotation=16, ha="right")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    for bars in (bars1, bars2):
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    fig.tight_layout()
    output = output_dir / "mux_method_accuracy_comparison.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    if args.slots < 2:
        raise ValueError("This comparison requires at least two signals")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    train_augmented, train_evaluated, test_base = build_comparison_datasets(
        args.data_dir, args.download
    )
    train_data, validation_data, test_data = split_datasets(
        train_augmented,
        train_evaluated,
        test_base,
        args.slots,
        args.seed,
        5000,
        False,
    )
    if args.training_size:
        if not 1 <= args.training_size <= len(train_data.base):
            raise ValueError("--training-size exceeds the available training split")
        train_data.base = Subset(train_data.base, range(args.training_size))
        train_data.set_epoch(0)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    results = [
        train_method(
            method,
            args,
            train_data,
            train_loader,
            validation_loader,
            test_loader,
            device,
        )
        for method in args.methods
    ]
    save_results(results, args.output_dir)
    figure_path = draw_figure(results, args.output_dir)

    print("\nMethod comparison on the official MNIST test split")
    print("method | mean accuracy | all three correct | parameters | images/s")
    for result in results:
        print(
            f"{result['display_name']} | "
            f"{100 * result['mean_accuracy']:.2f}% | "
            f"{100 * result['all_correct_accuracy']:.2f}% | "
            f"{result['parameters']} | {result['images_per_second']:.0f}"
        )
    print(f"Figure: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
