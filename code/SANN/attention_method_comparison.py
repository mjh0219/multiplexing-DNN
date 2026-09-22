import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DATASETS = {
    "susy": {
        "display_name": "UCI SUSY",
        "uci_id": 279,
        "uci_page": "https://archive.ics.uci.edu/dataset/279/susy",
        "download_url": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00279/SUSY.csv.gz"
        ),
        "cached_path": (
            "large_tabular_results/susy/susy_50000_sample.csv"
        ),
        "full_samples": 5000000,
        "features": 18,
        "classes": 2,
        "label_position": "first",
    },
    "covertype": {
        "display_name": "UCI Covertype",
        "uci_id": 31,
        "uci_page": "https://archive.ics.uci.edu/dataset/31/covertype",
        "download_url": (
            "http://archive.ics.uci.edu/ml/machine-learning-databases/"
            "covtype/covtype.data.gz"
        ),
        "cached_path": (
            "large_tabular_results/covertype/"
            "covertype_50000_sample.csv"
        ),
        "full_samples": 581012,
        "features": 54,
        "classes": 7,
        "label_position": "last",
    },
}


@dataclass
class SplitData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    input_dim: int
    num_classes: int
    used_samples: int


class FullyConnectedDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, second_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, second_dim),
            nn.BatchNorm1d(second_dim),
            nn.ReLU(),
            nn.Linear(second_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FeatureAttentionDNN(nn.Module):
    """A scalar attention gate is learned for every input feature."""

    def __init__(self, input_dim, hidden_dim, second_dim, num_classes):
        super().__init__()
        self.feature_logits = nn.Parameter(torch.zeros(input_dim))
        self.classifier = FullyConnectedDNN(
            input_dim, hidden_dim, second_dim, num_classes
        )

    def forward(self, x):
        gates = 2.0 * torch.sigmoid(self.feature_logits)
        return self.classifier(x * gates)


class SelfAttentionDNN(nn.Module):
    """Each scalar feature is treated as one token for multi-head attention."""

    def __init__(
        self,
        input_dim,
        token_dim,
        num_heads,
        hidden_dim,
        num_classes,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.value_projection = nn.Linear(1, token_dim)
        self.feature_embedding = nn.Parameter(
            torch.randn(1, input_dim, token_dim) * 0.02
        )
        self.attention = nn.MultiheadAttention(
            token_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(token_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * token_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        tokens = self.value_projection(x.unsqueeze(-1))
        tokens = tokens + self.feature_embedding
        attended, _ = self.attention(
            tokens, tokens, tokens, need_weights=False
        )
        attended = self.norm(tokens + attended)
        return self.classifier(attended.flatten(start_dim=1))


class StructuralLinear(nn.Module):
    """Linear layer with learned edge scores and a fixed post-selection mask."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.edge_logits = nn.Parameter(torch.zeros(output_dim, input_dim))
        self.register_buffer("edge_mask", torch.ones(output_dim, input_dim))
        self.selected = False
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        edge_weights = 2.0 * torch.sigmoid(self.edge_logits)
        effective_weight = self.weight * edge_weights
        if self.selected:
            effective_weight = effective_weight * self.edge_mask
        return nn.functional.linear(x, effective_weight, self.bias)

    def attention_penalty(self):
        return torch.sigmoid(self.edge_logits).mean()

    def select_edges(self, keep_ratio):
        """Keep an equal fan-in for every target neuron."""
        fan_in = self.weight.shape[1]
        keep_per_neuron = max(1, int(round(fan_in * keep_ratio)))
        importance = (
            self.weight.detach().abs()
            * torch.sigmoid(self.edge_logits.detach())
        )
        selected_indices = importance.topk(
            keep_per_neuron, dim=1, largest=True
        ).indices
        mask = torch.zeros_like(self.edge_mask)
        mask.scatter_(1, selected_indices, 1.0)
        self.edge_mask.copy_(mask)
        self.selected = True
        self.edge_logits.requires_grad_(False)

    def retained_parameters(self):
        return int(self.edge_mask.sum().item()) + self.bias.numel()


class StructuralAttentionDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, second_dim, num_classes):
        super().__init__()
        self.structural_1 = StructuralLinear(input_dim, hidden_dim)
        self.norm_1 = nn.BatchNorm1d(hidden_dim)
        self.structural_2 = StructuralLinear(hidden_dim, second_dim)
        self.norm_2 = nn.BatchNorm1d(second_dim)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(second_dim, num_classes)

    def forward(self, x):
        x = self.dropout(torch.relu(self.norm_1(self.structural_1(x))))
        x = torch.relu(self.norm_2(self.structural_2(x)))
        return self.output(x)

    def attention_penalty(self):
        return (
            self.structural_1.attention_penalty()
            + self.structural_2.attention_penalty()
        ) / 2.0

    def select_edges(self, keep_ratio):
        self.structural_1.select_edges(keep_ratio)
        self.structural_2.select_edges(keep_ratio)

    def retained_parameters(self):
        return (
            self.structural_1.retained_parameters()
            + self.structural_2.retained_parameters()
            + sum(p.numel() for p in self.norm_1.parameters())
            + sum(p.numel() for p in self.norm_2.parameters())
            + sum(p.numel() for p in self.output.parameters())
        )

    def retained_edges(self):
        return int(
            self.structural_1.edge_mask.sum().item()
            + self.structural_2.edge_mask.sum().item()
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset_dataframe(args):
    dataset = DATASETS[args.dataset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_candidates = [
        Path(args.local_csv) if args.local_csv else None,
        Path(dataset["cached_path"]),
        output_dir / f"{args.dataset}_{args.max_samples}_sample.csv",
    ]
    for candidate in cached_candidates:
        if candidate and candidate.exists():
            frame = pd.read_csv(candidate, nrows=args.max_samples)
            if args.local_csv or len(frame) >= args.max_samples:
                return frame, candidate.resolve(), False

    try:
        frame = pd.read_csv(
            dataset["download_url"],
            header=None,
            compression="gzip",
            nrows=args.max_samples,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{dataset['display_name']} could not be downloaded. Download "
            "the compressed CSV from its UCI page and pass the extracted "
            "CSV with --local-csv."
        ) from exc

    frame.columns = [f"col_{i}" for i in range(frame.shape[1])]
    local_path = output_dir / f"{args.dataset}_{len(frame)}_sample.csv"
    frame.to_csv(local_path, index=False)
    return frame, local_path.resolve(), True


def prepare_data(args):
    dataset = DATASETS[args.dataset]
    frame, source_path, downloaded = load_dataset_dataframe(args)
    expected_columns = dataset["features"] + 1
    if frame.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns (label + features), "
            f"got {frame.shape[1]}."
        )

    if dataset["label_position"] == "first":
        y = frame.iloc[:, 0].astype(np.int64).to_numpy()
        x = frame.iloc[:, 1:].astype(np.float32).to_numpy()
    else:
        y = frame.iloc[:, -1].astype(np.int64).to_numpy()
        x = frame.iloc[:, :-1].astype(np.float32).to_numpy()
    _, y = np.unique(y, return_inverse=True)
    y = y.astype(np.int64)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=args.seed,
        stratify=y_train_val,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    data = SplitData(
        x_train=torch.tensor(x_train),
        y_train=torch.tensor(y_train),
        x_val=torch.tensor(x_val),
        y_val=torch.tensor(y_val),
        x_test=torch.tensor(x_test),
        y_test=torch.tensor(y_test),
        input_dim=x.shape[1],
        num_classes=len(np.unique(y)),
        used_samples=len(frame),
    )
    return data, source_path, downloaded


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def make_loader(x, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_model(
    model,
    data,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    attention_penalty=0.0,
):
    loader = make_loader(
        data.x_train, data.y_train, batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            if attention_penalty and hasattr(model, "attention_penalty"):
                loss = loss + attention_penalty * model.attention_penalty()
            loss.backward()
            optimizer.step()
    return time.perf_counter() - start


def accuracy(model, x, y, batch_size=4096):
    loader = make_loader(x, y, batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            predictions = model(batch_x).argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.numel()
    return correct / total


def train_and_measure(name, model, data, args):
    training_time = train_model(
        model,
        data,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
    )
    return {
        "Method": name,
        "Training Time (s)": training_time,
        "Selection Time (s)": 0.0,
        "Total Time (s)": training_time,
        "NN Space Cost (parameters)": count_parameters(model),
        "Retained Edges": np.nan,
        "Accuracy": accuracy(model, data.x_test, data.y_test),
    }


def run_experiment(args):
    dataset = DATASETS[args.dataset]
    set_seed(args.seed)
    data, source_path, downloaded = prepare_data(args)
    rows = []

    set_seed(args.seed)
    rows.append(
        train_and_measure(
            "FC DNN",
            FullyConnectedDNN(
                data.input_dim,
                args.hidden_dim,
                args.second_dim,
                data.num_classes,
            ),
            data,
            args,
        )
    )

    set_seed(args.seed)
    rows.append(
        train_and_measure(
            "Feature Attention",
            FeatureAttentionDNN(
                data.input_dim,
                args.hidden_dim,
                args.second_dim,
                data.num_classes,
            ),
            data,
            args,
        )
    )

    set_seed(args.seed)
    rows.append(
        train_and_measure(
            "Self-Attention",
            SelfAttentionDNN(
                data.input_dim,
                args.token_dim,
                args.num_heads,
                args.hidden_dim,
                data.num_classes,
            ),
            data,
            args,
        )
    )

    set_seed(args.seed)
    sann = StructuralAttentionDNN(
        data.input_dim,
        args.hidden_dim,
        args.second_dim,
        data.num_classes,
    )
    sann_training_time = train_model(
        sann,
        data,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        attention_penalty=args.sparsity_penalty,
    )
    selection_start = time.perf_counter()
    sann.select_edges(args.keep_ratio)
    selection_time = time.perf_counter() - selection_start
    sann_training_time += train_model(
        sann,
        data,
        args.finetune_epochs,
        args.batch_size,
        args.finetune_learning_rate,
        args.weight_decay,
    )
    rows.append(
        {
            "Method": "Proposed SANN",
            "Training Time (s)": sann_training_time,
            "Selection Time (s)": selection_time,
            "Total Time (s)": sann_training_time + selection_time,
            "NN Space Cost (parameters)": sann.retained_parameters(),
            "Retained Edges": sann.retained_edges(),
            "Accuracy": accuracy(sann, data.x_test, data.y_test),
        }
    )

    results = pd.DataFrame(rows)
    metadata = {
        "dataset_key": args.dataset,
        "dataset": dataset["display_name"],
        "uci_dataset_id": dataset["uci_id"],
        "uci_page": dataset["uci_page"],
        "download_url": dataset["download_url"],
        "full_samples": dataset["full_samples"],
        "local_data": str(source_path),
        "downloaded_during_this_run": downloaded,
        "used_samples": data.used_samples,
        "features": data.input_dim,
        "classes": data.num_classes,
        "split": "60% training, 20% validation, 20% testing",
        "seed": args.seed,
        "epochs": args.epochs,
        "sann_finetune_epochs": args.finetune_epochs,
        "sann_keep_ratio": args.keep_ratio,
    }
    return results, metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare attention methods on a large UCI dataset."
    )
    parser.add_argument(
        "--dataset", choices=sorted(DATASETS), default="susy"
    )
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--local-csv", default="")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--second-dim", type=int, default=32)
    parser.add_argument("--token-dim", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--finetune-learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sparsity-penalty", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", default="attention_comparison_results"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results, metadata = run_experiment(args)

    results_path = output_dir / "attention_comparison_results.csv"
    metadata_path = output_dir / "metadata.json"

    results.to_csv(results_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(results.to_string(index=False))
    print(f"\nResults: {results_path.resolve()}")
    print(f"Dataset: {metadata['local_data']}")
    print(f"UCI page: {metadata['uci_page']}")


if __name__ == "__main__":
    main()
