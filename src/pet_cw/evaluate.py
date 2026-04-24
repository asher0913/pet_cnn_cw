"""Evaluate a saved checkpoint on the validation set and the test set.

Training already saves validation metrics, so the main reasons to
use this script are:

* to get fresh confusion matrices and per-class reports on the
  official test split, or
* to re-score a model after moving checkpoints between machines.

It loads whatever arguments the checkpoint recorded, so the split
and image size match the training run exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from pet_cw.data import build_dataloaders
from pet_cw.models import build_model
from pet_cw.train import run_one_epoch, run_tta_evaluation
from pet_cw.utils import (
    get_device,
    save_confusion_outputs,
    save_json,
    save_per_class_accuracy_bar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Oxford-IIIT Pet checkpoint on val and test splits.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt or last_model.pt.")
    parser.add_argument("--data-dir", default=None,
                        help="Override the data dir baked into the checkpoint.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to put fresh evaluation artifacts.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from training time.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true",
                        help="Download the dataset if it is missing.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Only re-score the validation split.")
    parser.add_argument("--tta", action="store_true",
                        help="Also evaluate the test split with horizontal-flip TTA "
                             "(averages logits over original + flipped). Recorded as "
                             "test_acc_tta alongside the plain test_acc.")
    return parser.parse_args()


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """torch.compile adds ``_orig_mod.`` prefixes to parameter keys.

    If a checkpoint was saved from a compiled model, the prefix has
    to be stripped before loading into an uncompiled model. Training
    already unwraps before saving, so this is really belt-and-braces.
    """
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    # weights_only=False because the checkpoint pickles a full dict
    # (args, indices, class_names). Safe here: the file was produced
    # by this project so the payload structure is trusted.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = SimpleNamespace(**checkpoint["args"])

    data_dir = args.data_dir or saved_args.data_dir
    batch_size = args.batch_size or saved_args.batch_size
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild loaders from the saved args so val split is identical.
    data = build_dataloaders(
        data_dir=data_dir,
        image_size=saved_args.image_size,
        batch_size=batch_size,
        val_fraction=saved_args.val_fraction,
        num_workers=args.num_workers,
        # Use the eval transform for both loaders; at inference time
        # augmentation randomness would only add noise to the numbers.
        augmentation="none",
        seed=saved_args.seed,
        download=args.download,
    )

    model_info = build_model(
        model_name=saved_args.model,
        num_classes=saved_args.num_classes,
        pretrained=False,         # weights come from the checkpoint, not ImageNet.
        freeze_backbone=False,    # doesn't matter for inference.
        dropout=getattr(saved_args, "dropout", 0.3),
    )
    model = model_info.model
    model.load_state_dict(strip_compile_prefix(checkpoint["model_state"]))

    device = get_device(args.device)
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss()

    # --- Validation ---
    with torch.no_grad():
        val_loss, val_acc, val_targets, val_predictions = run_one_epoch(
            model=model, loader=data.val_loader,
            criterion=criterion, optimizer=None,
            device=device, scaler=None, use_amp=False,
            grad_clip=0.0, epoch=int(checkpoint.get("epoch", 0)),
            phase="val(reeval)",
        )

    save_confusion_outputs(output_dir, val_targets, val_predictions,
                           checkpoint["class_names"], prefix="validation")
    save_per_class_accuracy_bar(output_dir, val_targets, val_predictions,
                                checkpoint["class_names"], prefix="validation")

    results: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_samples": len(val_targets),
    }

    # --- Test ---
    test_acc_tta: float | None = None
    if not args.skip_test:
        with torch.no_grad():
            test_loss, test_acc, test_targets, test_predictions = run_one_epoch(
                model=model, loader=data.test_loader,
                criterion=criterion, optimizer=None,
                device=device, scaler=None, use_amp=False,
                grad_clip=0.0, epoch=0, phase="test",
            )
        save_confusion_outputs(output_dir, test_targets, test_predictions,
                               checkpoint["class_names"], prefix="test")
        save_per_class_accuracy_bar(output_dir, test_targets, test_predictions,
                                    checkpoint["class_names"], prefix="test")
        results.update({
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_samples": len(test_targets),
        })

        if args.tta:
            tta_loss, test_acc_tta, tta_targets, tta_predictions = run_tta_evaluation(
                model=model, loader=data.test_loader,
                criterion=criterion,
                device=device, use_amp=False,
            )
            save_confusion_outputs(output_dir, tta_targets, tta_predictions,
                                   checkpoint["class_names"], prefix="test_tta")
            save_per_class_accuracy_bar(output_dir, tta_targets, tta_predictions,
                                        checkpoint["class_names"], prefix="test_tta")
            results.update({
                "test_loss_tta": float(tta_loss),
                "test_acc_tta": float(test_acc_tta),
            })

    save_json(output_dir / "metrics.json", results)

    print(f"Validation loss / acc: {val_loss:.4f} / {val_acc:.4f}")
    if not args.skip_test:
        print(f"Test loss / acc:       {test_loss:.4f} / {test_acc:.4f}")
    if test_acc_tta is not None:
        print(f"Test (TTA) loss / acc: {tta_loss:.4f} / {test_acc_tta:.4f}")
    print(f"Evaluation artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
