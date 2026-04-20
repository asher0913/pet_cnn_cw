"""Data loading and augmentation for the Oxford-IIIT Pet dataset.

A few design notes worth keeping in mind:

* The dataset ships two splits, ``trainval`` (3680 images) and ``test``
  (3669 images). Training uses a subset of ``trainval``, with a
  held-out validation slice out of ``trainval`` for model selection.
  The official ``test`` split is only touched at the very end. This
  avoids the common mistake of tuning against the test set, which
  inflates numbers.
* Train and validation transforms must differ (augmentation only on
  the train side), so two ``OxfordIIITPet`` datasets are built with
  the same seed and each is wrapped in a ``Subset`` that picks the
  right indices.
* ImageNet normalisation is used by default because the transfer
  learning Task 1 assumes it. The custom CNN tolerates it fine too;
  dataset-specific stats were tested and the improvement was inside
  noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Fallback in case torchvision does not expose `classes` for some version.
# Copied in the same alphabetical order torchvision uses internally.
OXFORD_PET_CLASSES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
    "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]

# Standard ImageNet statistics. Pretrained torchvision models were
# trained with these, so matching them is required for Task 1 to
# work well.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataBundle:
    """Container for the three loaders and the class names."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    train_indices: list[int]
    val_indices: list[int]


def build_transforms(image_size: int, augmentation: str) -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform).

    ``augmentation`` selects how aggressive the training-time augmentation
    is. The eval transform is always deterministic so validation and test
    numbers are reproducible.

    Augmentation choices come from the lecture on CNNs plus what tends
    to help for fine-grained animal classification (pets differ in
    subtle colour and texture cues, so colour jitter is kept mild).
    """

    augmentation = augmentation.lower()

    if augmentation == "none":
        train_steps: list[object] = [transforms.Resize((image_size, image_size))]

    elif augmentation == "basic":
        # Horizontal flip is safe for cats and dogs, they are bilaterally
        # symmetric. This alone gets most of the easy wins.
        train_steps = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

    elif augmentation == "strong":
        # RandomResizedCrop gives the biggest boost because pet photos
        # have very different framings. The scale range is kept tight
        # to reduce the chance of cropping out the animal entirely.
        train_steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.80, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
        ]

    else:
        raise ValueError("augmentation must be one of: none, basic, strong")

    train_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # RandomErasing goes after Normalize, otherwise the zero-valued
    # erased patch becomes a non-zero value in normalised space.
    if augmentation == "strong":
        train_steps.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)))

    eval_steps = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def get_class_names(dataset: datasets.OxfordIIITPet) -> list[str]:
    """Try the public attribute first, fall back to the private one.

    torchvision changed the attribute location between versions, so
    this helper keeps the code compatible with whatever ships on the
    lab machines.
    """
    classes = getattr(dataset, "classes", None)
    if classes:
        return list(classes)

    private_classes = getattr(dataset, "_CLASSES", None)
    if private_classes:
        return list(private_classes)

    return OXFORD_PET_CLASSES.copy()


def split_indices(length: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic train/val split of a dataset of ``length`` items.

    Using ``torch.Generator().manual_seed(seed)`` instead of numpy so
    the split matches regardless of which device runs the code.
    """
    if not 0.05 <= val_fraction <= 0.5:
        raise ValueError("val_fraction must be between 0.05 and 0.5")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(length, generator=generator).tolist()
    val_size = int(round(length * val_fraction))

    # Sort the indices inside each subset so dataloader workers read
    # files in a nicer order (less disk thrash on cold cache).
    val_indices = sorted(permutation[:val_size])
    train_indices = sorted(permutation[val_size:])
    return train_indices, val_indices


def build_dataloaders(
    data_dir: str | Path,
    image_size: int,
    batch_size: int,
    val_fraction: float,
    num_workers: int,
    augmentation: str,
    seed: int,
    download: bool,
) -> DataBundle:
    """Build train / val / test loaders in one call.

    The official ``test`` split is never seen during training. It is
    only used by ``evaluate.py`` or at the end of training for the
    final reported numbers.
    """
    data_dir = Path(data_dir)
    train_transform, eval_transform = build_transforms(image_size=image_size, augmentation=augmentation)

    # One download call does the work for all three dataset objects.
    base_dataset = datasets.OxfordIIITPet(
        root=str(data_dir),
        split="trainval",
        target_types="category",
        download=download,
        transform=None,
    )
    class_names = get_class_names(base_dataset)
    train_indices, val_indices = split_indices(len(base_dataset), val_fraction, seed)

    # Different transforms are needed for train vs val, hence two
    # dataset wrappers that point at the same files but do different
    # things.
    train_dataset = datasets.OxfordIIITPet(
        root=str(data_dir), split="trainval", target_types="category",
        download=False, transform=train_transform,
    )
    val_dataset = datasets.OxfordIIITPet(
        root=str(data_dir), split="trainval", target_types="category",
        download=False, transform=eval_transform,
    )
    test_dataset = datasets.OxfordIIITPet(
        root=str(data_dir), split="test", target_types="category",
        download=download, transform=eval_transform,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    pin_memory = torch.cuda.is_available()

    # persistent_workers keeps workers alive between epochs and speeds
    # up small datasets like this one noticeably. Only valid when
    # num_workers > 0.
    common_loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_subset, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        train_indices=train_indices,
        val_indices=val_indices,
    )


def build_image_transform(image_size: int) -> transforms.Compose:
    """Transform used at inference time on a single PIL image."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize_for_display(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse the ImageNet normalisation so matplotlib shows real colours.

    ``tensor`` is expected as [C, H, W] in normalised space. Output is
    clamped to [0, 1]. This is needed whenever a batch is plotted back
    out for inspection.
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


def label_names_from_indices(class_names: Sequence[str], labels: Sequence[int]) -> list[str]:
    return [class_names[int(label)] for label in labels]
