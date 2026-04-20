"""Shared helpers: metrics, plotting, checkpoint packaging, seeding.

Nothing here is specific to a particular model. Training, evaluation
and the visualisation scripts all pull from this module.
"""

from __future__ import annotations

import csv
import json
import random
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

import matplotlib

# "Agg" backend has no GUI dependency. Necessary on headless GPU boxes.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402


class AverageMeter:
    """Running mean that weighs by the number of items in each update.

    Simpler than a full TensorBoard scalar but good enough for logging
    per-epoch loss when batch sizes vary (last batch is often smaller).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG in use and, optionally, make cuDNN deterministic.

    Full determinism costs throughput (cudnn.benchmark=False), so by
    default only the RNGs are seeded and cuDNN is left free to pick
    the fastest kernels. The ``--deterministic`` flag is there for
    reproducibility runs when writing the report.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(requested: str) -> torch.device:
    """Resolve a device string. ``auto`` picks CUDA if available."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    """Return (correct, total) from a batch of logits and targets."""
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == targets).sum().item()
    return int(correct), int(targets.numel())


def namespace_to_dict(args: Namespace | dict[str, Any]) -> dict[str, Any]:
    """Make an argparse Namespace JSON-serialisable.

    Mostly needed because some args are ``Path`` objects, which json
    does not like by default.
    """
    if isinstance(args, Namespace):
        raw = vars(args)
    else:
        raw = args
    output: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value
    return output


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_history_csv(path: str | Path, history: list[dict[str, float]]) -> None:
    """Dump the per-epoch metrics list as a CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(path: str | Path, history: list[dict[str, float]]) -> None:
    """Two-panel plot: loss and accuracy across epochs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="Train accuracy")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="Val accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training / Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_confusion_outputs(
    output_dir: str | Path,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str],
    prefix: str = "validation",
) -> None:
    """Save the confusion matrix as CSV + heatmap PNG and the sklearn classification report.

    The heatmap is sized based on number of classes so labels stay
    readable. 37 classes on a default 10x8 figure ends up unreadable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(output_dir / f"{prefix}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    # Scale figure size with number of classes so 37 class names fit.
    fig_width = max(12, len(class_names) * 0.45)
    fig_height = max(10, len(class_names) * 0.38)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        cm, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=True, square=False, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{prefix.capitalize()} Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=220)
    plt.close(fig)

    report = classification_report(
        y_true, y_pred, labels=labels,
        target_names=class_names, zero_division=0,
    )
    (output_dir / f"{prefix}_classification_report.txt").write_text(report, encoding="utf-8")


def save_per_class_accuracy_bar(
    output_dir: str | Path,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str],
    prefix: str = "validation",
) -> None:
    """Horizontal bar chart of per-class accuracy.

    Makes it obvious at a glance which classes the model struggles on.
    Very handy for the Discussion section of the report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    accuracies = []
    for class_index in range(len(class_names)):
        mask = y_true_arr == class_index
        if mask.sum() == 0:
            accuracies.append(0.0)
        else:
            accuracies.append(float((y_pred_arr[mask] == class_index).mean()))

    # Sort so the worst classes are at the top of the plot.
    order = np.argsort(accuracies)
    sorted_names = [class_names[i] for i in order]
    sorted_values = [accuracies[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(6, len(class_names) * 0.25)))
    ax.barh(range(len(class_names)), sorted_values, color="steelblue")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_title(f"Per-Class Accuracy ({prefix})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_per_class_accuracy.png", dpi=200)
    plt.close(fig)


def checkpoint_payload(
    model_state: dict[str, torch.Tensor],
    args: Namespace,
    class_names: list[str],
    train_indices: list[int],
    val_indices: list[int],
    epoch: int,
    best_val_acc: float,
) -> dict[str, Any]:
    """Build the dict that gets saved into best_model.pt / last_model.pt.

    The CLI args and the train/val indices are stored alongside the
    weights so later evaluation runs can reproduce the same split
    without ambiguity.
    """
    return {
        "model_state": model_state,
        "args": namespace_to_dict(args),
        "class_names": class_names,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
    }
