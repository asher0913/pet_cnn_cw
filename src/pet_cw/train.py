"""Training entry point for both tasks.

The same training loop works for the custom CNN and the transfer
learning models; what differs is how the model is built and whether
differential learning rates are enabled. Everything is parameterised
via CLI args so one script can drive every experiment in the ablation.

What happens in one run:

1. Seed and device setup.
2. Build dataloaders (train / val / test).
3. Build the model and its optimiser. Transfer learning models get
   parameter groups with different LRs for backbone and head.
4. Loop for ``--epochs``. After each epoch, log to history.csv and
   refresh training_curves.png. Whenever val acc improves,
   best_model.pt and the confusion-matrix artifacts for that epoch
   are saved.
5. Once training is done, load the best model and run it on the
   official test split to produce the final number the report
   should quote.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm

from pet_cw.data import build_dataloaders
from pet_cw.models import MODEL_CHOICES, build_model, build_param_groups, count_parameters
from pet_cw.utils import (
    AverageMeter,
    accuracy_from_logits,
    checkpoint_payload,
    get_device,
    plot_training_curves,
    save_confusion_outputs,
    save_json,
    save_per_class_accuracy_bar,
    seed_everything,
    write_history_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models for the COMP3065 pet-classification coursework.")

    # --- Experiment book-keeping ---
    parser.add_argument("--experiment-name", default="pet_experiment",
                        help="Used to name the output directory.")
    parser.add_argument("--data-dir", default="./data", help="Dataset root.")
    parser.add_argument("--output-dir", default="./outputs",
                        help="Where run artifacts land.")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Write to output-dir/experiment-name without appending a timestamp.")

    # --- Model ---
    parser.add_argument("--model", default="custom", choices=MODEL_CHOICES,
                        help="Model architecture to train.")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use ImageNet weights for transfer models. Ignored for custom.")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze the pretrained feature extractor (feature-extraction strategy).")
    parser.add_argument("--num-classes", type=int, default=37,
                        help="Oxford-IIIT Pet has 37 breeds.")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout on the final FC layer (custom CNN only).")

    # --- Data ---
    parser.add_argument("--image-size", type=int, default=160,
                        help="160 for custom, 224 for transfer models.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--augmentation", choices=["none", "basic", "strong"], default="basic")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--download", action="store_true",
                        help="Download the dataset if it is missing.")

    # --- Optimiser and schedule ---
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Only used by SGD.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler",
                        choices=["none", "step", "cosine", "plateau"],
                        default="none")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Cross-entropy label smoothing factor. 0.1 is a reasonable starting point.")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Max grad norm. 0 disables clipping.")
    parser.add_argument("--head-lr-mult", type=float, default=10.0,
                        help="Multiplier on the new classifier head's LR for transfer fine-tuning.")

    # --- System ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true",
                        help="Force deterministic cuDNN (slower).")
    parser.add_argument("--device", default="auto",
                        help="auto | cuda | cuda:0 | cpu.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training on CUDA.")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap the model in torch.compile (needs PyTorch 2.x).")
    parser.add_argument("--test-at-end", action="store_true",
                        help="Run the best model on the official test split once training finishes.")
    return parser.parse_args()


def create_run_dir(output_dir: str | Path, experiment_name: str, no_timestamp: bool) -> Path:
    """Make an output directory. Timestamps prevent earlier runs from being overwritten."""
    base = Path(output_dir)
    if no_timestamp:
        run_dir = base / experiment_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / f"{experiment_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> Optimizer:
    """Build an optimiser over trainable parameters.

    For transfer models, parameter groups are constructed so the
    backbone and the new head can have different LRs.
    """
    groups = build_param_groups(
        model=model,
        model_name=args.model,
        base_lr=args.lr,
        head_lr_multiplier=args.head_lr_mult,
    )

    if args.optimizer == "adam":
        return torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    # SGD
    return torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


def build_scheduler(args: argparse.Namespace, optimizer: Optimizer):
    """Map the CLI choice to a torch.optim.lr_scheduler object."""
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        # One step-down at 1/3 of training, another at 2/3.
        return StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def unwrap_compiled(model: nn.Module) -> nn.Module:
    """If the model was wrapped with torch.compile, return the original.

    Saving the compiled state_dict leaves ``_orig_mod.`` prefixes in
    the keys, which confuses loading later. Raw module weights are
    always saved instead.
    """
    return getattr(model, "_orig_mod", model)


def run_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    grad_clip: float,
    epoch: int,
    phase: str,
) -> tuple[float, float, list[int], list[int]]:
    """Run one pass over ``loader`` in either train or eval mode.

    ``optimizer=None`` means evaluation-only: no grads, no weight
    update. The same function is used from evaluate.py to score a
    checkpoint.
    """
    is_train = optimizer is not None
    model.train(mode=is_train)

    loss_meter = AverageMeter()
    correct_total = 0
    sample_total = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(loader, desc=f"{phase} epoch {epoch}", leave=False)
    for images, targets in progress:
        # non_blocking helps when pinned memory is set on the loader.
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            # New torch.amp API (PyTorch 2.1+). Works on CPU too, it just
            # becomes a no-op when use_amp is False.
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                if scaler is not None and use_amp:
                    # AMP path: scale the loss so small fp16 grads don't underflow.
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in model.parameters() if p.requires_grad),
                            max_norm=grad_clip,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in model.parameters() if p.requires_grad),
                            max_norm=grad_clip,
                        )
                    optimizer.step()

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)

        correct, total = accuracy_from_logits(logits.detach(), targets)
        correct_total += correct
        sample_total += total

        predictions = torch.argmax(logits.detach(), dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

        progress.set_postfix(
            loss=f"{loss_meter.average:.4f}",
            acc=f"{correct_total / max(sample_total, 1):.4f}",
        )

    accuracy = correct_total / max(sample_total, 1)
    return loss_meter.average, accuracy, all_targets, all_predictions


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    device = get_device(args.device)

    # Mixed precision only makes sense on CUDA. AMP on CPU runs fp32.
    use_amp = bool(args.amp and device.type == "cuda")
    run_dir = create_run_dir(args.output_dir, args.experiment_name, args.no_timestamp)

    # Friendly nudge, not an error: pretrained nets do their best work
    # at 224 because that is what ImageNet training used.
    if args.model != "custom" and args.image_size != 224:
        print("Note: transfer models usually want 224x224 input. "
              f"Running with {args.image_size} instead.")

    data = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        augmentation=args.augmentation,
        seed=args.seed,
        download=args.download,
    )

    model_info = build_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=bool(args.pretrained and args.model != "custom"),
        freeze_backbone=bool(args.freeze_backbone and args.model != "custom"),
        dropout=args.dropout,
    )
    model = model_info.model.to(device)

    if args.compile and hasattr(torch, "compile"):
        # torch.compile gives a nice speedup on recent GPUs but can add
        # noise to the first few iterations while it warms up.
        model = torch.compile(model)

    total_params, trainable_params = count_parameters(model)
    print(f"Run directory:    {run_dir}")
    print(f"Device:           {device}")
    print(f"Classes:          {len(data.class_names)}")
    print(f"Train / val size: {len(data.train_indices)} / {len(data.val_indices)}")
    print(f"Test size:        {len(data.test_loader.dataset)}")
    print(f"Model:            {args.model}  pretrained={model_info.pretrained}  "
          f"freeze={model_info.freeze_backbone}")
    print(f"Parameters:       total={total_params:,}  trainable={trainable_params:,}")

    save_json(
        run_dir / "run_config.json",
        {
            "args": vars(args),
            "class_names": data.class_names,
            "train_size": len(data.train_indices),
            "val_size": len(data.val_indices),
            "test_size": len(data.test_loader.dataset),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        },
    )

    # Label smoothing is a very cheap regulariser that helps on
    # fine-grained problems. It pushes the model away from putting
    # all probability mass on one class, which reduces overfitting.
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    # New torch.amp API. The old torch.cuda.amp.* is deprecated.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_one_epoch(
            model=model, loader=data.train_loader,
            criterion=criterion, optimizer=optimizer,
            device=device, scaler=scaler, use_amp=use_amp,
            grad_clip=args.grad_clip,
            epoch=epoch, phase="train",
        )

        with torch.no_grad():
            val_loss, val_acc, val_targets, val_predictions = run_one_epoch(
                model=model, loader=data.val_loader,
                criterion=criterion, optimizer=None,
                device=device, scaler=None, use_amp=use_amp,
                grad_clip=0.0,
                epoch=epoch, phase="val",
            )

        # Plateau scheduler needs a metric. Others just step per epoch.
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(current_lr),
        }
        history.append(row)
        write_history_csv(run_dir / "history.csv", history)
        plot_training_curves(run_dir / "training_curves.png", history)

        payload = checkpoint_payload(
            model_state=unwrap_compiled(model).state_dict(),
            args=args,
            class_names=data.class_names,
            train_indices=data.train_indices,
            val_indices=data.val_indices,
            epoch=epoch,
            best_val_acc=max(best_val_acc, val_acc),
        )
        # last_model.pt is always overwritten; best_model.pt only when val acc goes up.
        torch.save(payload, last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(payload, best_path)
            save_confusion_outputs(run_dir, val_targets, val_predictions,
                                   data.class_names, prefix="validation")
            save_per_class_accuracy_bar(run_dir, val_targets, val_predictions,
                                        data.class_names, prefix="validation")
            print(f"Epoch {epoch}: new best val acc = {val_acc:.4f}")

        print(f"Epoch {epoch:03d}/{args.epochs:03d}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  lr={current_lr:.6f}")

    # Optional final evaluation on the untouched test split.
    # The best checkpoint is always restored first so numbers match
    # the model actually being reported in the writeup.
    test_acc: float | None = None
    if args.test_at_end:
        print("\nEvaluating the best checkpoint on the test split...")
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        unwrap_compiled(model).load_state_dict(checkpoint["model_state"])
        with torch.no_grad():
            test_loss, test_acc, test_targets, test_predictions = run_one_epoch(
                model=model, loader=data.test_loader,
                criterion=criterion, optimizer=None,
                device=device, scaler=None, use_amp=use_amp,
                grad_clip=0.0, epoch=0, phase="test",
            )
        save_confusion_outputs(run_dir, test_targets, test_predictions,
                               data.class_names, prefix="test")
        save_per_class_accuracy_bar(run_dir, test_targets, test_predictions,
                                    data.class_names, prefix="test")
        print(f"Test loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    summary = {
        "best_val_acc": best_val_acc,
        "best_model": str(best_path),
    }
    if test_acc is not None:
        summary["test_acc"] = float(test_acc)
    save_json(run_dir / "summary.json", summary)

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")
    if test_acc is not None:
        print(f"Test accuracy (held-out split): {test_acc:.4f}")
    print(f"All artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
