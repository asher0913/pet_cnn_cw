"""COMP3065 Oxford-IIIT Pet classification coursework.

Two models live here: a custom CNN trained from scratch (Task 2)
and a wrapper around torchvision pretrained networks for transfer
learning (Task 1). The training code, evaluation, Grad-CAM and
prediction visualisation all share the same data pipeline.
"""

__all__ = ["data", "models", "utils", "gradcam", "visualize"]
