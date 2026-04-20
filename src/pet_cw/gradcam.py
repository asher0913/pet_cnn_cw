"""Grad-CAM for explaining what the CNN looks at.

A from-scratch implementation of Selvaraju et al. 2017
(Gradient-weighted Class Activation Mapping). The steps are:

1. Pick a convolutional layer late in the network. This is the
   layer whose activations should carry high-level semantic
   meaning but still retain useful spatial resolution.
2. Register a forward hook to capture its activations ``A``.
3. Register a backward hook to capture gradients ``dY/dA`` for
   the chosen class score.
4. Channel-wise weights ``alpha_k`` are the spatial average of
   those gradients. Intuitively, a channel that contributes a lot
   to the class score will have large gradients spread across its
   spatial positions.
5. The class-activation map is ``ReLU(sum_k alpha_k * A_k)``. The
   ReLU keeps only the positive contributions, which is what
   Grad-CAM calls the discriminative region.
6. Bilinearly upsample the map to the original image size and
   overlay it on the RGB image as a heatmap.

The CW brief specifically points at the pytorch-cnn-visualizations
repo as a resource; this implementation is in the same spirit but
written from scratch line by line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from pet_cw.data import denormalize_for_display


class GradCAM:
    """Run Grad-CAM on a model given a target layer.

    Usage::

        cam = GradCAM(model, model.layer4[-1])
        heatmap = cam(image_tensor, class_index=3)

    Call :meth:`remove_hooks` after use to let Python free the
    references. If the object is garbage-collected this also
    happens automatically, but explicit cleanup is easier to
    reason about.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Forward hook: store the output of ``target_layer``.
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        # Backward hook: store the gradient flowing out of ``target_layer``.
        # register_full_backward_hook replaces the deprecated one.
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    # --- Hook callbacks ---

    def _save_activations(self, _module: nn.Module, _inputs, output: torch.Tensor) -> None:
        # Detach so the autograd graph is not kept alive by this handle.
        self.activations = output.detach()

    def _save_gradients(self, _module: nn.Module, _grad_in, grad_out) -> None:
        # grad_out is a tuple; the first element is the gradient of the
        # loss with respect to the layer's output, which is what Grad-CAM
        # needs.
        self.gradients = grad_out[0].detach()

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    # --- Main computation ---

    def __call__(self, image: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, int, float]:
        """Return (heatmap, predicted_class, predicted_probability).

        ``image`` must be a [1, C, H, W] tensor already on the model's device.
        """
        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError("GradCAM expects a single image with shape [1, C, H, W].")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(image)
        probabilities = F.softmax(logits, dim=1)

        if class_index is None:
            class_index = int(torch.argmax(logits, dim=1).item())

        # Scalar to back-propagate through: the chosen class logit.
        score = logits[0, class_index]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients were not captured. "
                               "Did the target layer run during forward?")

        # alpha_k = mean over spatial dims of the gradient for channel k.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation channels. shape: [1, 1, h, w]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to the input image size so the heatmap can be overlaid.
        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)

        # Normalise to [0, 1] for display.
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        heatmap = cam.squeeze().cpu().numpy()
        probability = float(probabilities[0, class_index].item())
        return heatmap, class_index, probability


def overlay_heatmap(image_tensor: torch.Tensor, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a [H, W] heatmap on a normalised [C, H, W] image.

    The image is de-normalised first (otherwise the colours look
    wrong), the heatmap is run through a matplotlib colour map,
    and the two are blended with transparency ``alpha``. Output is
    an RGB float array in [0, 1] ready for ``plt.imshow``.
    """
    image = denormalize_for_display(image_tensor).permute(1, 2, 0).cpu().numpy()

    colour_map = plt.get_cmap("jet")
    coloured_heatmap = colour_map(heatmap)[..., :3]  # drop alpha channel

    blended = (1.0 - alpha) * image + alpha * coloured_heatmap
    return np.clip(blended, 0.0, 1.0)


def save_gradcam_grid(
    output_path: str | Path,
    samples: Iterable[tuple[torch.Tensor, int, str]],
    model: nn.Module,
    target_layer: nn.Module,
    class_names: list[str],
    device: torch.device,
    ncols: int = 4,
) -> None:
    """Run Grad-CAM on a batch of images and save a grid figure.

    Each item in ``samples`` is (image_tensor, true_label, caption).
    The figure has two columns per sample: original image, and
    original + heatmap overlay. Helpful for the Discussion section.
    """
    samples = list(samples)
    if not samples:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cam = GradCAM(model, target_layer)

    # Two subplots per sample: raw image and overlay. Arrange side by side.
    rows = (len(samples) + ncols - 1) // ncols
    fig, axes = plt.subplots(rows, ncols * 2, figsize=(ncols * 4.5, rows * 3.0))
    axes = np.atleast_2d(axes)

    for idx, (image_tensor, true_label, caption) in enumerate(samples):
        row = idx // ncols
        col = (idx % ncols) * 2

        image_on_device = image_tensor.unsqueeze(0).to(device)
        heatmap, pred_idx, prob = cam(image_on_device)
        overlay = overlay_heatmap(image_tensor, heatmap, alpha=0.4)

        axes[row, col].imshow(denormalize_for_display(image_tensor).permute(1, 2, 0).cpu().numpy())
        axes[row, col].axis("off")
        axes[row, col].set_title(
            f"true: {class_names[true_label]}",
            fontsize=9,
        )

        axes[row, col + 1].imshow(overlay)
        axes[row, col + 1].axis("off")
        status = "OK" if pred_idx == true_label else "WRONG"
        axes[row, col + 1].set_title(
            f"{status}: {class_names[pred_idx]} ({prob:.2f})",
            fontsize=9,
            color=("green" if pred_idx == true_label else "red"),
        )

    # Hide any trailing empty subplots.
    total_slots = axes.shape[0] * axes.shape[1]
    used_slots = len(samples) * 2
    for leftover in range(used_slots, total_slots):
        r = leftover // axes.shape[1]
        c = leftover % axes.shape[1]
        axes[r, c].axis("off")

    if caption:
        fig.suptitle(caption, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    cam.remove_hooks()
