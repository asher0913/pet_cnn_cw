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
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
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


class ModelEma:
    """Exponential moving average (EMA) of model parameters and buffers.

    Classic Polyak averaging (Polyak and Juditsky, 1992) that keeps a
    shadow copy of the network weights, updated after every optimiser
    step as::

        shadow := eff_decay * shadow + (1 - eff_decay) * current

    where ``eff_decay`` is the *bias-corrected* decay (see below).

    Why the bias correction matters
    -------------------------------

    A naive ``decay = 0.999`` schedule is the formula that gets
    quoted most often in tutorials, but it has a pitfall when the
    model is trained from scratch: the shadow is *initialised* with
    whatever weights the model starts with, which for Task 2 are the
    random Kaiming init. If that shadow is then updated with decay =
    0.999 for 3680 steps (80 epochs x 46 batches), the final shadow
    still retains ``exp(-3680 * 0.001) approx 2.5 percent`` of the
    original random initialisation mixed in with the trained
    weights. On a small fine-grained dataset like Oxford-IIIT Pet
    this contamination is enough to visibly regress the EMA
    checkpoint below the raw training checkpoint - exactly what we
    observed empirically with the first, un-corrected version of
    this class.

    The standard fix, used by timm, fastai and the EfficientNet
    reference code, is a *decay warmup*::

        eff_decay = min(decay, (1 + n) / (10 + n))

    where ``n`` is the number of updates. The effect is to let the
    shadow track the current weights aggressively for the first few
    iterations (``n = 0`` gives ``eff_decay = 0.1``, so ``shadow``
    jumps almost entirely to ``current``), then smoothly tighten
    towards the nominal ``decay`` as training progresses. The
    contamination from the random init is flushed out within the
    first couple of hundred steps.

    Notes on edge cases
    -------------------

    * BatchNorm tracks two floating-point buffers (``running_mean``
      and ``running_var``); those are EMA-averaged in the normal way.
    * BatchNorm also tracks ``num_batches_tracked`` as ``int64``.
      That buffer is copied verbatim rather than averaged, since a
      weighted mean of integer step counts makes no sense.
    * ``state_dict`` (not ``named_parameters``) is the iteration
      target so both learnable weights and buffers get shadowed.
      This matches what timm does and is important for BN.
    """

    # Warmup factor in the bias-correction schedule. 10 is the value
    # used by timm / EfficientNet / fastai; it gives eff_decay = 0.91
    # at step 100, 0.99 at step 989 and plateaus near the nominal
    # ``decay`` for the rest of training.
    BIAS_WARMUP = 10

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        # Clone on-device so per-step updates are in-place GPU ops.
        # Memory overhead is the model's weight footprint, which is a
        # few tens of MB even for ResNet-18 - trivial on the GPUs this
        # project targets.
        self.shadow: dict[str, torch.Tensor] = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Apply one EMA update step using ``model``'s current weights.

        The effective decay is the bias-corrected schedule described
        in the class docstring. Each call bumps ``num_updates`` first
        so the very first update uses ``eff_decay = 2 / 11 approx
        0.18`` rather than zero.
        """
        self.num_updates += 1
        eff_decay = min(
            self.decay,
            (1.0 + self.num_updates) / (self.BIAS_WARMUP + self.num_updates),
        )
        for name, tensor in model.state_dict().items():
            shadow_tensor = self.shadow[name]
            if tensor.dtype.is_floating_point:
                shadow_tensor.mul_(eff_decay).add_(
                    tensor.detach(), alpha=1.0 - eff_decay,
                )
            else:
                # Integer buffers (num_batches_tracked) just get copied.
                shadow_tensor.copy_(tensor)

    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Temporarily swap EMA weights into ``model``.

        Returns the original state dict so the caller can restore the
        training weights after validation is done.
        """
        backup = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        model.load_state_dict(self.shadow)
        return backup

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Reverse a prior :meth:`apply_to` call."""
        model.load_state_dict(backup)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow


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
    # Default is False so that the run-config.json saved by a Task 2
    # run cannot be mistaken for a pretrained-weights run. Transfer
    # experiments must pass ``--pretrained`` explicitly.
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction,
                        default=False,
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
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear LR warmup for the first N epochs. Only combines with cosine; "
                             "ignored for other schedulers. 0 disables warmup.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Cross-entropy label smoothing factor. 0.1 is a reasonable starting point.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Beta distribution parameter for Mixup. 0 disables Mixup; 0.2 is typical.")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Max grad norm. 0 disables clipping.")
    parser.add_argument("--head-lr-mult", type=float, default=10.0,
                        help="Multiplier on the new classifier head's LR for transfer fine-tuning.")
    parser.add_argument("--min-lr", type=float, default=0.0,
                        help="Cosine scheduler LR floor (passed as eta_min). The default 0 "
                             "is what CosineAnnealingLR already does; set to e.g. 1e-5 to keep "
                             "the last few epochs making small but useful updates instead of "
                             "coasting at zero LR.")
    parser.add_argument("--ema", action="store_true",
                        help="Maintain an EMA of model weights and use it for validation and "
                             "checkpointing (Polyak averaging). Standard in modern recipes.")
    parser.add_argument("--ema-decay", type=float, default=0.9999,
                        help="Nominal EMA decay (the asymptotic value). The effective decay "
                             "used per step is bias-corrected via min(decay, (1+n)/(10+n)) "
                             "so the shadow is not contaminated by the random init during "
                             "the first few hundred updates - see ModelEma for details. "
                             "0.9999 matches the timm / EfficientNet default; 0.999 is fine "
                             "for very short runs too.")
    parser.add_argument("--tta", action="store_true",
                        help="Apply horizontal-flip test-time augmentation during --test-at-end. "
                             "Averages logits from the original image and its horizontal flip; "
                             "adds nothing to training cost and typically +0.5-1.5 points on test.")

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
    """Map the CLI choice to a torch.optim.lr_scheduler object.

    Cosine annealing optionally gets a short linear warmup wrapped
    around it via ``SequentialLR``. Warmup helps when training from
    scratch because BatchNorm statistics are still garbage for the
    first few iterations and a full learning rate can make the first
    epoch diverge. For transfer learning with pretrained weights the
    warmup is less important but does not hurt.
    """
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        # One step-down at 1/3 of training, another at 2/3.
        return StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    if args.scheduler == "cosine":
        warmup = max(int(args.warmup_epochs), 0)
        cosine_epochs = max(args.epochs - warmup, 1)
        # eta_min keeps the LR floor above zero in the cosine tail. Leaving
        # it at zero wastes the last few epochs - CosineAnnealingLR hits
        # exactly zero for the final step and the model coasts. A tiny
        # floor like 1e-5 lets those epochs keep refining the weights.
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=float(args.min_lr))
        if warmup == 0:
            return cosine
        # Start at start_factor * base_lr and linearly ramp to base_lr
        # by the end of the warmup window. 0.1 is a standard choice.
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
        return SequentialLR(optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup])
    if args.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return (mixed_images, targets_a, targets_b, lam).

    Mixup (Zhang et al., 2018) builds a virtual training example by
    convex-combining two real images and their labels::

        x_mix = lam * x_a + (1 - lam) * x_b
        y_mix = lam * y_a + (1 - lam) * y_b

    Because PyTorch's ``CrossEntropyLoss`` expects integer targets,
    the combined loss is computed in the training loop as
    ``lam * loss_a + (1 - lam) * loss_b`` instead of mixing labels
    directly, which is numerically identical for cross-entropy.

    ``lam`` is drawn from a Beta(alpha, alpha) distribution; small
    alpha concentrates mass near 0 or 1 (little mixing), large alpha
    concentrates mass near 0.5 (heavy mixing). 0.2 works well here.
    """
    if alpha <= 0.0:
        # Defensive: caller should check, but this keeps the function safe.
        return images, targets, targets, 1.0

    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[permutation]
    return mixed, targets, targets[permutation], lam


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
    mixup_alpha: float = 0.0,
    ema: ModelEma | None = None,
) -> tuple[float, float, list[int], list[int]]:
    """Run one pass over ``loader`` in either train or eval mode.

    ``optimizer=None`` means evaluation-only: no grads, no weight
    update. The same function is used from evaluate.py to score a
    checkpoint.

    When ``mixup_alpha > 0`` and ``optimizer is not None``, Mixup is
    applied to each training batch. Validation and test phases never
    see Mixup, so their numbers stay directly comparable across runs.
    Training accuracy in Mixup mode is reported against the primary
    target of each mixed pair; the number is a bit noisier than
    plain training accuracy but still tracks model progress.
    """
    is_train = optimizer is not None
    use_mixup = is_train and mixup_alpha > 0.0
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

        # Apply Mixup before the forward pass. Do this outside autocast
        # so the interpolation happens in fp32 and does not introduce
        # rounding noise on top of the mixing.
        if use_mixup:
            mixed_images, targets_a, targets_b, lam = mixup_batch(images, targets, mixup_alpha)
        else:
            mixed_images, targets_a, targets_b, lam = images, targets, targets, 1.0

        with torch.set_grad_enabled(is_train):
            # New torch.amp API (PyTorch 2.1+). Works on CPU too, it just
            # becomes a no-op when use_amp is False.
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(mixed_images)
                if use_mixup:
                    # Cross-entropy is linear in the one-hot target, so
                    # mixing the losses is identical to mixing the labels.
                    loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
                else:
                    loss = criterion(logits, targets_a)

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

                # EMA update runs after every parameter step. If the AMP
                # scaler skipped this step due to a non-finite gradient
                # the model's weights are unchanged and the update is
                # effectively a no-op - safe to call unconditionally.
                if ema is not None:
                    ema.update(unwrap_compiled(model))

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


def run_tta_evaluation(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate with horizontal-flip test-time augmentation.

    The model runs forward twice per batch: once on the batch as-is,
    once on the batch flipped left-to-right. The two logit tensors are
    averaged before taking argmax. Cats and dogs are bilaterally
    symmetric so flipping does not create unrealistic views; the flip
    pair exposes the model to a second "look" at the same animal and
    averaging reduces prediction variance.

    TTA is strictly a test-time technique - training is untouched - so
    it does not change the architecture or add parameters, and it
    therefore stays inside the coursework's permitted methods.
    """
    model.eval()
    loss_meter = AverageMeter()
    correct_total = 0
    sample_total = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="test (TTA)", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits_main = model(images)
                logits_flip = model(torch.flip(images, dims=[3]))
                logits = (logits_main + logits_flip) * 0.5
                loss = criterion(logits, targets)

            loss_meter.update(loss.item(), targets.size(0))
            predictions = torch.argmax(logits, dim=1)
            correct_total += (predictions == targets).sum().item()
            sample_total += targets.size(0)
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

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

    # EMA is built against the unwrapped model so its state-dict keys
    # match those we later load into the model (torch.compile adds an
    # _orig_mod. prefix that would otherwise have to be stripped).
    ema: ModelEma | None = None
    if args.ema:
        ema = ModelEma(unwrap_compiled(model), decay=args.ema_decay)
        print(f"EMA enabled with decay={args.ema_decay}. "
              f"Validation and best_model.pt will use the EMA weights.")

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_model.pt"

    # Wall-clock timing so the report can quote training cost per model.
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, _, _ = run_one_epoch(
            model=model, loader=data.train_loader,
            criterion=criterion, optimizer=optimizer,
            device=device, scaler=scaler, use_amp=use_amp,
            grad_clip=args.grad_clip,
            epoch=epoch, phase="train",
            mixup_alpha=args.mixup_alpha,
            ema=ema,
        )

        # When EMA is active we run validation twice: once on the raw
        # training weights, once on the EMA shadow. The EMA number is
        # the one used for model selection (because the saved
        # checkpoint is the EMA state), but the raw number is logged
        # too so the report can diagnose any EMA/raw divergence and
        # so a failed EMA run is immediately visible in history.csv
        # rather than hiding behind a single merged metric.
        val_loss_raw: float | None = None
        val_acc_raw: float | None = None
        if ema is not None:
            with torch.no_grad():
                val_loss_raw, val_acc_raw, _, _ = run_one_epoch(
                    model=model, loader=data.val_loader,
                    criterion=criterion, optimizer=None,
                    device=device, scaler=None, use_amp=use_amp,
                    grad_clip=0.0,
                    epoch=epoch, phase="val(raw)",
                )

        ema_backup: dict[str, torch.Tensor] | None = None
        if ema is not None:
            ema_backup = ema.apply_to(unwrap_compiled(model))
        try:
            with torch.no_grad():
                val_loss, val_acc, val_targets, val_predictions = run_one_epoch(
                    model=model, loader=data.val_loader,
                    criterion=criterion, optimizer=None,
                    device=device, scaler=None, use_amp=use_amp,
                    grad_clip=0.0,
                    epoch=epoch, phase="val",
                )
        finally:
            if ema is not None and ema_backup is not None:
                ema.restore(unwrap_compiled(model), ema_backup)
        epoch_seconds = time.time() - epoch_start

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
            "epoch_seconds": float(epoch_seconds),
        }
        # When EMA is active, also log the raw-weight validation
        # numbers so the report can show EMA vs raw side-by-side.
        # Keys are absent for non-EMA runs so history.csv stays
        # narrow in that case.
        if ema is not None:
            row["val_loss_raw"] = float(val_loss_raw) if val_loss_raw is not None else 0.0
            row["val_acc_raw"] = float(val_acc_raw) if val_acc_raw is not None else 0.0
        history.append(row)
        write_history_csv(run_dir / "history.csv", history)
        plot_training_curves(run_dir / "training_curves.png", history)

        # When EMA is active the canonical weights for evaluation are
        # the shadow ones, so we persist those as ``model_state`` and
        # evaluate.py / visualize.py load them transparently without
        # needing to know EMA was used. The raw training weights are
        # also saved to ``best_model_raw.pt`` whenever EMA is on, as
        # a safety fallback: if the EMA weights underperform (we have
        # seen this when Mixup is combined with a heavily-averaged
        # shadow on this small dataset), the raw checkpoint is
        # available for re-evaluation without retraining.
        if ema is not None:
            weights_for_checkpoint = ema.state_dict()
        else:
            weights_for_checkpoint = unwrap_compiled(model).state_dict()

        payload = checkpoint_payload(
            model_state=weights_for_checkpoint,
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
            # Sibling raw-weight checkpoint: same payload format but
            # with the training weights instead of the EMA shadow.
            if ema is not None:
                raw_payload = checkpoint_payload(
                    model_state=unwrap_compiled(model).state_dict(),
                    args=args,
                    class_names=data.class_names,
                    train_indices=data.train_indices,
                    val_indices=data.val_indices,
                    epoch=epoch,
                    best_val_acc=max(best_val_acc, val_acc),
                )
                torch.save(raw_payload, run_dir / "best_model_raw.pt")
            save_confusion_outputs(run_dir, val_targets, val_predictions,
                                   data.class_names, prefix="validation")
            save_per_class_accuracy_bar(run_dir, val_targets, val_predictions,
                                        data.class_names, prefix="validation")
            print(f"Epoch {epoch}: new best val acc = {val_acc:.4f}")

        if ema is not None and val_acc_raw is not None:
            print(f"Epoch {epoch:03d}/{args.epochs:03d}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} (ema)  "
                  f"val_acc_raw={val_acc_raw:.4f}  "
                  f"lr={current_lr:.6f}  time={epoch_seconds:.1f}s")
        else:
            print(f"Epoch {epoch:03d}/{args.epochs:03d}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
                  f"lr={current_lr:.6f}  time={epoch_seconds:.1f}s")

    total_training_seconds = time.time() - training_start
    mean_epoch_seconds = (
        sum(row["epoch_seconds"] for row in history) / len(history) if history else 0.0
    )

    # Optional final evaluation on the untouched test split.
    # The best checkpoint is always restored first so numbers match
    # the model actually being reported in the writeup.
    test_acc: float | None = None
    test_acc_tta: float | None = None
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

        # TTA evaluation reuses the exact same checkpoint - the single
        # number that changes is how predictions are aggregated at
        # test time. Both numbers are reported so the report can
        # quantify the TTA contribution independently of EMA.
        if args.tta:
            print("Running test-time augmentation (horizontal flip average)...")
            tta_loss, test_acc_tta, tta_targets, tta_predictions = run_tta_evaluation(
                model=model, loader=data.test_loader,
                criterion=criterion,
                device=device, use_amp=use_amp,
            )
            save_confusion_outputs(run_dir, tta_targets, tta_predictions,
                                   data.class_names, prefix="test_tta")
            save_per_class_accuracy_bar(run_dir, tta_targets, tta_predictions,
                                        data.class_names, prefix="test_tta")
            print(f"Test (TTA) loss={tta_loss:.4f}  test_acc_tta={test_acc_tta:.4f}  "
                  f"(delta vs no TTA: {(test_acc_tta - test_acc) * 100:+.2f} pp)")

    # summary.json collects every number the report is likely to cite,
    # so the aggregation script does not have to re-open each history.csv.
    summary = {
        "experiment_name": args.experiment_name,
        "model": args.model,
        "pretrained": bool(args.pretrained and args.model != "custom"),
        "freeze_backbone": bool(args.freeze_backbone and args.model != "custom"),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "learning_rate": args.lr,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "augmentation": args.augmentation,
        "ema": bool(args.ema),
        "ema_decay": float(args.ema_decay) if args.ema else 0.0,
        "min_lr": float(args.min_lr),
        "tta": bool(args.tta),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "best_val_acc": float(best_val_acc),
        "best_model": str(best_path),
        "mean_epoch_seconds": float(mean_epoch_seconds),
        "total_training_seconds": float(total_training_seconds),
    }
    if test_acc is not None:
        summary["test_acc"] = float(test_acc)
    if test_acc_tta is not None:
        summary["test_acc_tta"] = float(test_acc_tta)
    save_json(run_dir / "summary.json", summary)

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")
    if test_acc is not None:
        print(f"Test accuracy (held-out split): {test_acc:.4f}")
    if test_acc_tta is not None:
        print(f"Test accuracy with TTA:         {test_acc_tta:.4f}")
    print(f"Mean epoch time: {mean_epoch_seconds:.1f}s  "
          f"(total training: {total_training_seconds/60:.1f} min)")
    print(f"All artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
