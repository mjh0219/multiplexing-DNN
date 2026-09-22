#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("mnist", "cifar10", "cifar100"), default="cifar10"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./mux_results"))
    parser.add_argument("--method", choices=("baseline", "mux", "both"), default="both")
    parser.add_argument(
        "--architecture",
        choices=("mnist-cnn", "small-cnn", "resnet18"),
        default="resnet18",
        help="Use mnist-cnn for MNIST, small-cnn for a CIFAR laptop test, or resnet18.",
    )
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument(
        "--variable-combinations",
        action="store_true",
        help="Train with random nonempty slot subsets and test every possible subset.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64, help="Number of image groups per batch.")
    parser.add_argument(
        "--validation-size",
        type=int,
        default=5000,
        help="Number of development images reserved for validation.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument("--download", dest="download", action="store_true")
    download_group.add_argument("--no-download", dest="download", action="store_false")
    parser.set_defaults(download=True)
    parser.add_argument("--quick-run", action="store_true", help="Run one epoch with ten batches.")
    parser.add_argument("--smoke-test", action="store_true", help="Test tensor shapes without data.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def mux_images(images: torch.Tensor) -> torch.Tensor:
    """Pack [B, S, C, H, W] into X_mul=[B, S*C, H, W]."""
    if images.ndim != 5:
        raise ValueError("images must have shape [batch, slots, channels, height, width]")
    batch, slots, channels, height, width = images.shape
    return images.contiguous().view(batch, slots * channels, height, width)


def demux_images(x_mul: torch.Tensor, slots: int) -> torch.Tensor:
    """Recover [B, S, C, H, W] from a channel-multiplexed tensor."""
    if x_mul.ndim != 4 or x_mul.shape[1] % slots != 0:
        raise ValueError("X_mul channels must be divisible by the number of slots")
    batch, packed_channels, height, width = x_mul.shape
    return x_mul.contiguous().view(batch, slots, packed_channels // slots, height, width)


def mux_outputs(logits: torch.Tensor) -> torch.Tensor:
    """Pack [B, S, C] logits into Y_mul=[B, S*C]."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, slots, classes]")
    return logits.contiguous().view(logits.shape[0], -1)


def demux_outputs(y_mul: torch.Tensor, slots: int, num_classes: int) -> torch.Tensor:
    """Recover the per-slot class logits [B, S, C] from Y_mul."""
    expected = slots * num_classes
    if y_mul.ndim != 2 or y_mul.shape[1] != expected:
        raise ValueError("Y_mul has the wrong number of output values")
    return y_mul.contiguous().view(y_mul.shape[0], slots, num_classes)


def mux_labels(labels: torch.Tensor) -> torch.Tensor:
    """Pack target labels [B, S] into one ordered target vector."""
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, slots]")
    return labels.contiguous().view(-1)


def demux_labels(y_mul: torch.Tensor, slots: int) -> torch.Tensor:
    """Recover the per-slot target labels [B, S]."""
    if y_mul.ndim != 1 or y_mul.numel() % slots != 0:
        raise ValueError("multiplexed labels must be divisible by the number of slots")
    return y_mul.contiguous().view(-1, slots)


class RandomImageGroups(Dataset):
    """Create ordered groups of randomly paired images.

    Slot order is retained, so output slot s always corresponds to input slot s.
    Test groups are deterministic to make results repeatable.
    """

    def __init__(
        self,
        base: Dataset,
        slots: int,
        seed: int,
        training: bool,
        variable_combinations: bool = False,
        exhaustive_combinations: bool = False,
    ) -> None:
        if slots < 1:
            raise ValueError("slots must be at least 1")
        self.base = base
        self.slots = slots
        self.seed = seed
        self.training = training
        self.variable_combinations = variable_combinations
        self.exhaustive_combinations = exhaustive_combinations
        self.epoch = 0
        self._permutation = []
        self._combination_ids = []
        self.set_epoch(0)

    def __len__(self) -> int:
        groups = len(self.base) // self.slots
        if self.variable_combinations and self.exhaustive_combinations:
            return groups * (2**self.slots - 1)
        return groups

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        effective_epoch = epoch if self.training else 0
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * effective_epoch)
        self._permutation = torch.randperm(len(self.base), generator=generator).tolist()
        groups = len(self.base) // self.slots
        combination_count = 2**self.slots - 1
        balanced = torch.arange(groups) % combination_count + 1
        order = torch.randperm(groups, generator=generator)
        self._combination_ids = balanced[order].tolist()

    def _indices(self, index: int) -> Iterable[int]:
        start = index * self.slots
        return self._permutation[start : start + self.slots]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        combination_count = 2**self.slots - 1
        if self.variable_combinations and self.exhaustive_combinations:
            group_index = index // combination_count
            combination_id = index % combination_count + 1
        else:
            group_index = index
            combination_id = (
                self._combination_ids[group_index]
                if self.variable_combinations
                else combination_count
            )

        samples = [self.base[j] for j in self._indices(group_index)]
        images = torch.stack([sample[0] for sample in samples], dim=0)
        labels = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
        active = torch.tensor(
            [(combination_id >> slot) & 1 for slot in range(self.slots)],
            dtype=torch.bool,
        )
        images = images * active[:, None, None, None]
        return images, labels, active


def build_datasets(name: str, root: Path, slots: int, seed: int, download: bool):
    if name == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_augmented = datasets.MNIST(
            root=root, train=True, transform=transform, download=download
        )
        train_evaluated = datasets.MNIST(
            root=root, train=True, transform=transform, download=download
        )
        test_base = datasets.MNIST(
            root=root, train=False, transform=transform, download=download
        )
        return train_augmented, train_evaluated, test_base, 10, 1, 28

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )
    dataset_class = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    num_classes = 10 if name == "cifar10" else 100
    train_augmented = dataset_class(
        root=root, train=True, transform=train_transform, download=download
    )
    train_evaluated = dataset_class(
        root=root, train=True, transform=test_transform, download=download
    )
    test_base = dataset_class(root=root, train=False, transform=test_transform, download=download)
    return train_augmented, train_evaluated, test_base, num_classes, 3, 32


def split_datasets(
    train_augmented: Dataset,
    train_evaluated: Dataset,
    test_base: Dataset,
    slots: int,
    seed: int,
    validation_size: int,
    variable_combinations: bool,
):
    if not 0 < validation_size < len(train_augmented):
        raise ValueError("--validation-size must be between 1 and the training-set size")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_augmented), generator=generator).tolist()
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    train_base = Subset(train_augmented, training_indices)
    validation_base = Subset(train_evaluated, validation_indices)
    train_groups = RandomImageGroups(
        train_base,
        slots=slots,
        seed=seed,
        training=True,
        variable_combinations=variable_combinations,
    )
    validation_groups = RandomImageGroups(
        validation_base,
        slots=slots,
        seed=seed + 1,
        training=False,
        variable_combinations=variable_combinations,
        exhaustive_combinations=variable_combinations,
    )
    test_groups = RandomImageGroups(
        test_base,
        slots=slots,
        seed=seed + 2,
        training=False,
        variable_combinations=variable_combinations,
        exhaustive_combinations=variable_combinations,
    )
    return train_groups, validation_groups, test_groups


class CifarResNet18(nn.Module):
    """ResNet-18 adapted to 32x32 images and configurable input/output sizes."""

    def __init__(self, in_channels: int, output_dim: int) -> None:
        super().__init__()
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CifarSmallCNN(nn.Module):
    """Compact CNN for a fast, non-SANN multiplexing proof of concept."""

    def __init__(self, in_channels: int, output_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(1)
        return self.classifier(features)


class MnistCNN(nn.Module):
    """Small CNN that retains spatial information for handwritten digits."""

    def __init__(self, in_channels: int, output_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_backbone(architecture: str, in_channels: int, output_dim: int) -> nn.Module:
    if architecture == "mnist-cnn":
        return MnistCNN(in_channels, output_dim)
    if architecture == "small-cnn":
        return CifarSmallCNN(in_channels, output_dim)
    return CifarResNet18(in_channels, output_dim)


class MultiplexedClassifier(nn.Module):
    def __init__(
        self,
        slots: int,
        num_classes: int,
        input_channels: int = 3,
        architecture: str = "resnet18",
    ) -> None:
        super().__init__()
        self.slots = slots
        self.num_classes = num_classes
        self.backbone = build_backbone(
            architecture, input_channels * slots, slots * num_classes
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x_mul = mux_images(images)
        y_mul = self.backbone(x_mul)
        return demux_outputs(y_mul, self.slots, self.num_classes)


class IndependentClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        architecture: str = "resnet18",
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(architecture, input_channels, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, slots, channels, height, width = images.shape
        flat_images = images.view(batch * slots, channels, height, width)
        flat_logits = self.backbone(flat_images)
        return flat_logits.view(batch, slots, -1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if active is None:
        return F.cross_entropy(logits.flatten(0, 1), mux_labels(labels))
    return F.cross_entropy(logits[active], labels[active])


def combination_name(active_row: torch.Tensor) -> str:
    slots = [str(index + 1) for index, enabled in enumerate(active_row) if enabled]
    return "_".join(slots)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    max_batches: Optional[int],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_predictions = 0
    total_correct = 0
    joint_correct = 0
    total_groups = 0
    slot_correct = None
    slot_total = None
    combination_correct: Dict[str, int] = {}
    combination_total: Dict[str, int] = {}
    combination_joint_correct: Dict[str, int] = {}
    combination_groups: Dict[str, int] = {}

    sync_device(device)
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels, active = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            active = active.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = classification_loss(logits, labels, active)
            if training:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(labels) & active
            if slot_correct is None:
                slot_correct = torch.zeros(labels.shape[1], dtype=torch.long)
                slot_total = torch.zeros(labels.shape[1], dtype=torch.long)
            slot_correct += correct.sum(dim=0).cpu()
            slot_total += active.sum(dim=0).cpu()
            active_count = int(active.sum().item())
            total_loss += float(loss.item()) * active_count
            total_correct += int(correct.sum().item())
            group_correct = (correct | ~active).all(dim=1)
            joint_correct += int(group_correct.sum().item())
            total_predictions += active_count
            total_groups += labels.shape[0]

            for row in range(labels.shape[0]):
                name = combination_name(active[row].detach().cpu())
                row_active = int(active[row].sum().item())
                combination_correct[name] = combination_correct.get(name, 0) + int(
                    correct[row].sum().item()
                )
                combination_total[name] = combination_total.get(name, 0) + row_active
                combination_joint_correct[name] = combination_joint_correct.get(
                    name, 0
                ) + int(group_correct[row].item())
                combination_groups[name] = combination_groups.get(name, 0) + 1

    sync_device(device)
    elapsed = time.perf_counter() - started
    result = {
        "loss": total_loss / max(total_predictions, 1),
        "mean_image_accuracy": total_correct / max(total_predictions, 1),
        "all_slots_accuracy": joint_correct / max(total_groups, 1),
        "all_active_accuracy": joint_correct / max(total_groups, 1),
        "elapsed_seconds": elapsed,
        "images_per_second": total_predictions / max(elapsed, 1e-9),
    }
    if slot_correct is not None and slot_total is not None:
        for slot, (correct_count, prediction_count) in enumerate(
            zip(slot_correct.tolist(), slot_total.tolist()), start=1
        ):
            result[f"slot_{slot}_accuracy"] = correct_count / max(prediction_count, 1)
    for name in sorted(combination_total):
        result[f"combination_{name}_accuracy"] = (
            combination_correct[name] / combination_total[name]
        )
        result[f"combination_{name}_all_correct"] = (
            combination_joint_correct[name] / combination_groups[name]
        )
    return result


def train_method(
    method: str,
    train_data: RandomImageGroups,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    train_loader: DataLoader,
    num_classes: int,
    input_channels: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    if method == "mux":
        model = MultiplexedClassifier(
            args.slots, num_classes, input_channels, args.architecture
        )
    else:
        model = IndependentClassifier(num_classes, input_channels, args.architecture)
    model.to(device)

    if args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    epochs = 1 if args.quick_run else args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    max_batches = 10 if args.quick_run else None
    best_accuracy = -1.0
    train_seconds = 0.0
    history = []

    for epoch in range(epochs):
        train_data.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, device, optimizer, max_batches)
        validation_metrics = run_epoch(
            model, validation_loader, device, None, max_batches
        )
        scheduler.step()
        train_seconds += train_metrics["elapsed_seconds"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["mean_image_accuracy"],
                "validation_accuracy": validation_metrics["mean_image_accuracy"],
                "validation_all_slots_accuracy": validation_metrics["all_slots_accuracy"],
            }
        )
        print(
            f"[{method}] epoch {epoch + 1:03d}/{epochs:03d} "
            f"train_acc={train_metrics['mean_image_accuracy']:.4f} "
            f"val_acc={validation_metrics['mean_image_accuracy']:.4f} "
            f"val_all_{args.slots}_correct={validation_metrics['all_slots_accuracy']:.4f}"
        )
        if validation_metrics["mean_image_accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["mean_image_accuracy"]
            checkpoint = {
                "method": method,
                "dataset": args.dataset,
                "slots": args.slots,
                "variable_combinations": args.variable_combinations,
                "num_classes": num_classes,
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "validation_metrics": validation_metrics,
            }
            torch.save(checkpoint, args.output_dir / f"best_{method}.pt")

    checkpoint = torch.load(args.output_dir / f"best_{method}.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    final_metrics = run_epoch(model, test_loader, device, None, max_batches)
    result = {
        "method": (
            f"Independent {args.architecture}"
            if method == "baseline"
            else f"MUX {args.architecture}"
        ),
        "dataset": args.dataset.upper(),
        "architecture": args.architecture,
        "optimizer": args.optimizer,
        "slots": args.slots,
        "variable_combinations": args.variable_combinations,
        "parameters": count_parameters(model),
        "training_seconds": train_seconds,
        "best_epoch": checkpoint["epoch"],
        **final_metrics,
    }
    with (args.output_dir / f"history_{method}.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return result


def save_results(results, output_dir: Path) -> None:
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    fieldnames = sorted({key for row in results for key in row})
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def preview(vector: torch.Tensor, values: int = 5) -> str:
    numbers = vector.detach().cpu().flatten()[:values].tolist()
    return "[" + ", ".join(f"{number:+.4f}" for number in numbers) + "]"


def smoke_test(
    slots: int = 3,
    num_classes: int = 10,
    seed: int = 42,
    architecture: str = "resnet18",
    input_channels: int = 3,
    image_size: int = 32,
) -> None:
    images = torch.randn(2, slots, input_channels, image_size, image_size)
    x_mul = mux_images(images)
    recovered = demux_images(x_mul, slots)
    assert torch.equal(images, recovered), "Input MUX/DEMUX must be exactly invertible"
    print("PASS: input DEMUX exactly recovered every value in x1, x2, and x3")

    labels = torch.randint(num_classes, (2, slots))
    packed_labels = mux_labels(labels)
    assert torch.equal(labels, demux_labels(packed_labels, slots))
    print("PASS: label DEMUX exactly recovered y1, y2, and y3")

    model = MultiplexedClassifier(slots, num_classes, input_channels, architecture)
    logits = model(images)
    y_mul = mux_outputs(logits)
    recovered_logits = demux_outputs(y_mul, slots, num_classes)
    assert torch.equal(logits, recovered_logits), "Output MUX/DEMUX must be exactly invertible"
    print("PASS: output DEMUX exactly recovered all three sets of class logits")
    loss = classification_loss(logits, labels)
    loss.backward()
    baseline = IndependentClassifier(num_classes, input_channels, architecture)
    baseline_logits = baseline(images)
    assert baseline_logits.shape == logits.shape
    print(f"PASS: multiplexed {architecture} completed one forward and backward pass")
    print("All smoke tests passed")
    print(f"random seed: {seed}")
    print(f"images: {tuple(images.shape)}")
    print(f"X_mul:  {tuple(x_mul.shape)}")
    print(f"Y_mul:  {tuple(y_mul.shape)}")
    print(f"outputs: {tuple(recovered_logits.shape)}")
    print("\nFirst random group (short vector previews):")
    for slot in range(slots):
        print(f"x{slot + 1} first 5 values: {preview(images[0, slot])}")
    print(f"X_mul first 15 values: {preview(x_mul[0], values=15)}")
    print(f"true class indices: {labels[0].tolist()}")
    for slot in range(slots):
        print(
            f"y_hat{slot + 1} first 5 logits: "
            f"{preview(recovered_logits[0, slot])}"
        )
    print(f"predicted class indices: {recovered_logits[0].argmax(dim=-1).tolist()}")
    print(f"maximum input recovery error: {(images - recovered).abs().max().item():.1f}")
    print(
        "maximum output recovery error: "
        f"{(logits - recovered_logits).abs().max().item():.1f}"
    )


def main() -> None:
    args = parse_args()
    if args.slots < 1:
        raise ValueError("--slots must be at least 1")
    if args.variable_combinations and args.slots > 10:
        raise ValueError(
            "Exhaustive variable-combination evaluation is limited to 10 slots; "
            "omit --variable-combinations for a signal-count sweep."
        )
    set_seed(args.seed)
    if args.smoke_test:
        is_mnist = args.dataset == "mnist"
        smoke_test(
            args.slots,
            seed=args.seed,
            architecture=args.architecture,
            input_channels=1 if is_mnist else 3,
            image_size=28 if is_mnist else 32,
        )
        return

    if args.dataset == "mnist" and args.architecture != "mnist-cnn":
        raise ValueError("Use --architecture mnist-cnn with --dataset mnist")
    if args.dataset != "mnist" and args.architecture == "mnist-cnn":
        raise ValueError("--architecture mnist-cnn is only available for MNIST")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    (
        train_augmented,
        train_evaluated,
        test_base,
        num_classes,
        input_channels,
        _image_size,
    ) = build_datasets(
        args.dataset, args.data_dir, args.slots, args.seed, args.download
    )
    train_data, validation_data, test_data = split_datasets(
        train_augmented,
        train_evaluated,
        test_base,
        args.slots,
        args.seed,
        args.validation_size,
        args.variable_combinations,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_kwargs)
    validation_loader = DataLoader(
        validation_data, shuffle=False, drop_last=False, **loader_kwargs
    )
    test_loader = DataLoader(test_data, shuffle=False, drop_last=False, **loader_kwargs)

    methods = ("baseline", "mux") if args.method == "both" else (args.method,)
    results = []
    for method in methods:
        results.append(
            train_method(
                method,
                train_data,
                validation_loader,
                test_loader,
                train_loader,
                num_classes,
                input_channels,
                args,
                device,
            )
        )
    save_results(results, args.output_dir)
    print(f"Saved experiment outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
