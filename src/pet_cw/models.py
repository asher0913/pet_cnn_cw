"""Model definitions for the coursework.

Two model families live here:

1. ``PetResNet`` (Task 2). Custom CNN built from scratch. The design
   borrows the residual-block idea from He et al. 2016 (ResNet) and
   the channel-attention idea from Hu et al. 2018 (Squeeze-and-Excite).
   The exact layer counts, channel widths, stem, and placement of
   the SE blocks are original. The network is deliberately small:
   the Oxford-IIIT Pet dataset only has around 3.7k training images
   after the validation split, so a very deep net would overfit.

2. Transfer-learning wrappers around torchvision models for Task 1.
   The final classifier is swapped for a 37-way layer and the feature
   extractor can be frozen for pure feature extraction or left
   trainable for full fine-tuning.

A helper at the bottom builds parameter groups with different learning
rates, which is the textbook way to fine-tune a pretrained backbone:
low LR on the backbone so ImageNet features are preserved, higher
LR on the fresh classifier head so the new weights can catch up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torchvision import models


TRANSFER_MODELS = {
    "resnet18", "resnet34", "resnet50",
    "vgg16", "vgg19",
    "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
}

MODEL_CHOICES = ["custom", *sorted(TRANSFER_MODELS)]


@dataclass(frozen=True)
class ModelInfo:
    """Bundle returned by :func:`build_model`."""

    model: nn.Module
    model_name: str
    pretrained: bool
    freeze_backbone: bool
    num_classes: int


# ---------------------------------------------------------------------------
# Building blocks for PetResNet
# ---------------------------------------------------------------------------


class SqueezeExcite(nn.Module):
    """Channel attention block from Hu et al. 2018.

    The feature map is squeezed with a global average pool, passed
    through a tiny MLP, and the sigmoid output is used to rescale
    each channel. Intuition: the network learns to amplify the
    channels that matter for the current input and attenuate the
    rest. For fine-grained pet classification, different breeds
    care about different channels (fur texture vs. ear shape vs.
    colour), so this is a natural fit.

    ``reduction`` is the bottleneck ratio; 16 is the value in the paper.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        # [B, C, H, W] -> [B, C] after GAP
        squeeze = self.pool(x).view(batch, channels)
        excitation = self.fc(squeeze).view(batch, channels, 1, 1)
        return x * excitation


class BasicResBlock(nn.Module):
    """A two-conv residual block with optional stride-2 downsampling.

    Conv -> BN -> ReLU -> Conv -> BN -> (+ shortcut) -> ReLU

    If in_channels != out_channels or stride != 1, the shortcut uses a
    1x1 conv so the addition actually lines up. This is the classic
    BasicBlock from the ResNet paper; the deeper Bottleneck variant
    is overkill for a dataset this size, so the simpler form is kept.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        # bias=False because BatchNorm absorbs the bias term.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut path. Identity is cheaper, but a 1x1 conv is needed
        # when the shapes change, otherwise the + op would fail.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual addition before the final ReLU (original ResNet).
        out = out + identity
        return self.relu(out)


# ---------------------------------------------------------------------------
# PetResNet: the custom CNN (Task 2)
# ---------------------------------------------------------------------------


class PetResNet(nn.Module):
    """Custom CNN for 37-way pet classification.

    Layout (for a 160x160 input):

        Stem:       3  -> 32   (3x3 conv, no stride)         160x160
        Stage 1:    32 -> 64   (ResBlock s=2, ResBlock s=1)   80x80
        Stage 2:    64 -> 128  (ResBlock s=2, ResBlock, SE)   40x40
        Stage 3:    128-> 256  (ResBlock s=2, ResBlock, SE)   20x20
        Tail:       GAP -> Dropout -> Linear(256, 37)

    Why this shape:

    * A 3x3 stride-1 stem keeps as much spatial detail as possible,
      which helps on fine-grained classes (for example, distinguishing
      the stripe pattern of a Bengal from an Egyptian Mau).
    * Downsampling is done inside the first ResBlock of each stage
      using stride-2 on the 3x3 conv. That is cleaner than a separate
      MaxPool and matches ResNet practice.
    * SE blocks are placed in the deeper stages only. Shallow features
      (edges, corners) do not benefit much from channel attention,
      but higher-level semantic channels do.
    * Global average pooling (instead of flatten + big FC) keeps the
      parameter count low and generalises better. The final FC has
      only ``256 * 37 + 37`` = 9509 parameters.

    Parameter count is around 2.7M. On Oxford-IIIT Pet that sits in
    the sweet spot: enough capacity to learn, few enough to train
    from scratch in a handful of dozen epochs without overfitting
    hard.
    """

    def __init__(self, num_classes: int = 37, dropout: float = 0.3) -> None:
        super().__init__()

        # Stem: gentle feature extraction at full resolution.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1: first downsample is here, not in the stem.
        self.stage1 = nn.Sequential(
            BasicResBlock(32, 64, stride=2),
            BasicResBlock(64, 64, stride=1),
        )

        # Stage 2: add SE after the feature maps are deep enough for
        # channel attention to be meaningful.
        self.stage2 = nn.Sequential(
            BasicResBlock(64, 128, stride=2),
            BasicResBlock(128, 128, stride=1),
            SqueezeExcite(128, reduction=16),
        )

        # Stage 3: final feature stage before global pooling.
        self.stage3 = nn.Sequential(
            BasicResBlock(128, 256, stride=2),
            BasicResBlock(256, 256, stride=1),
            SqueezeExcite(256, reduction=16),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout only on the classifier. BN already acts as a light
        # regulariser in the conv stack, so extra dropout there tended
        # to hurt in initial experiments.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """He initialisation for conv and linear, standard for BN."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Small init for the final classifier; biases at zero.
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        return self.classifier(x)

    def gradcam_target_layer(self) -> nn.Module:
        """Layer whose activations drive the Grad-CAM computation.

        The conv just before global pooling carries the highest-level
        spatial information, which is what Grad-CAM wants.
        """
        return self.stage3[-1]  # the final SE block sits on top of stage3


# ---------------------------------------------------------------------------
# Transfer learning wrappers (Task 1)
# ---------------------------------------------------------------------------


_WEIGHTS_TABLE = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet34": models.ResNet34_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
    "vgg16": models.VGG16_Weights.DEFAULT,
    "vgg19": models.VGG19_Weights.DEFAULT,
    "mobilenet_v2": models.MobileNet_V2_Weights.DEFAULT,
    "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.DEFAULT,
    "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
}


def _weights_for(model_name: str, pretrained: bool):
    """Return the torchvision Weights enum for a model name."""
    return _WEIGHTS_TABLE[model_name] if pretrained else None


def _freeze_all_parameters(model: nn.Module) -> None:
    """Freeze every parameter so only the replaced classifier trains.

    This is what Lab 6 called the ``feature extraction`` strategy:
    the pretrained network is treated as a fixed feature extractor.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False


def _replace_classifier(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    """Swap the final layer of a torchvision model for the 37-way head.

    Different families put the classifier in different places, so the
    replacement dispatches on the model name. The input feature count
    is read off the original layer so the code stays correct even if
    torchvision changes the default width.
    """
    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name.startswith("vgg"):
        # VGG stores the classifier as a Sequential, final layer is the last one.
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported transfer model: {model_name}")


def _classifier_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the new classifier layer so the caller can target it
    for differential learning rates."""
    if model_name.startswith("resnet"):
        return model.fc
    if model_name.startswith("vgg") or model_name.startswith("mobilenet"):
        return model.classifier[-1]
    raise ValueError(f"Unsupported transfer model: {model_name}")


def build_model(
    model_name: str,
    num_classes: int = 37,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
) -> ModelInfo:
    """Build the requested model and prepare it for training.

    For ``model_name == "custom"`` the return value wraps
    :class:`PetResNet`. Otherwise a torchvision model is loaded,
    optionally with pretrained weights, optionally with the backbone
    frozen, and the classifier is swapped for a ``num_classes``-way
    layer.
    """
    model_name = model_name.lower()

    if model_name == "custom":
        model = PetResNet(num_classes=num_classes, dropout=dropout)
        return ModelInfo(
            model=model,
            model_name=model_name,
            pretrained=False,
            freeze_backbone=False,
            num_classes=num_classes,
        )

    if model_name not in TRANSFER_MODELS:
        raise ValueError(f"Unknown model {model_name}. Valid choices: {', '.join(MODEL_CHOICES)}")

    weights = _weights_for(model_name, pretrained)
    factory = getattr(models, model_name)
    model = factory(weights=weights)

    # Freeze before replacing the classifier so the new head stays trainable.
    if freeze_backbone:
        _freeze_all_parameters(model)

    model = _replace_classifier(model, model_name=model_name, num_classes=num_classes)

    return ModelInfo(
        model=model,
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        num_classes=num_classes,
    )


# ---------------------------------------------------------------------------
# Parameter-group helper for differential learning rates
# ---------------------------------------------------------------------------


def build_param_groups(
    model: nn.Module,
    model_name: str,
    base_lr: float,
    head_lr_multiplier: float = 10.0,
) -> list[dict]:
    """Split parameters into backbone and head groups with different LRs.

    When full fine-tuning a pretrained network, the backbone weights
    are already good. Updating them with the same LR as the freshly
    initialised classifier tends to wash out those features early in
    training. The common trick is to give the classifier head a
    learning rate roughly 10x the backbone.

    For non-transfer models this returns a single group with ``base_lr``.
    """

    if model_name == "custom" or model_name not in TRANSFER_MODELS:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr}]

    head = _classifier_layer(model, model_name)
    head_param_ids = {id(p) for p in head.parameters()}

    head_params = [p for p in head.parameters() if p.requires_grad]
    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in head_param_ids
    ]

    # If the backbone is frozen only the head trains, and there is
    # nothing to contrast against, so the multiplier should not apply.
    # In the full fine-tune case the head gets a larger LR so the
    # fresh weights can catch up without destroying the pretrained ones.
    if not backbone_params:
        if not head_params:
            raise RuntimeError("No trainable parameters, check freeze_backbone and classifier replacement.")
        return [{"params": head_params, "lr": base_lr}]

    groups: list[dict] = [{"params": backbone_params, "lr": base_lr}]
    if head_params:
        groups.append({"params": head_params, "lr": base_lr * head_lr_multiplier})
    return groups


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def iter_trainable(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)
