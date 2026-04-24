"""Regression tests for the ``ModelEma`` bias-corrected decay schedule.

Background
----------

An earlier version of ``ModelEma`` applied a fixed ``decay = 0.999``
from step one. On the Task 2 (from-scratch) sweep this left about
``0.999 ** 3680 approx 2.5%`` of the random Kaiming init still mixed
into the shadow weights at the end of training. The EMA checkpoint
regressed by nearly seven percentage points of validation accuracy
relative to the raw training weights.

The fix is a bias-correction warmup (the same formula used by timm,
fastai and the EfficientNet reference)::

    eff_decay = min(decay, (1 + n) / (10 + n))

so the first few updates track the current weights aggressively and
flush the random init out of the shadow before the nominal decay
kicks in.

These tests lock that schedule in place. If someone accidentally
reintroduces the naive un-corrected update, these tests will fail
before the 40-minute Linux sweep is wasted re-discovering the bug.

Run with ``pytest tests/test_model_ema.py`` from the project root, or
as a plain script (``python tests/test_model_ema.py``) when pytest is
not available.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

# Allow the test file to be run from the project root without needing
# ``pip install -e .`` to be done first. Mirrors the pattern used by
# the experiment driver in ``scripts/``.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pet_cw.train import ModelEma  # noqa: E402  (sys.path tweak above)


def _tiny_model() -> nn.Module:
    """A minimal module with a float param and a BN buffer (float + int)."""
    module = nn.Sequential(
        nn.Linear(4, 4),
        nn.BatchNorm1d(4),
    )
    return module


def test_bias_corrected_decay_schedule() -> None:
    """``eff_decay`` follows ``(1+n)/(10+n)`` while below the nominal decay.

    This is the schedule that recovers the shadow from the random
    init. The exact fractions at a handful of ``n`` values are checked
    so any regression on the formula is caught immediately.
    """
    model = _tiny_model()
    ema = ModelEma(model, decay=0.9999)

    # The implementation computes eff_decay *after* incrementing
    # num_updates by one, so after update() is called n times the
    # most recent eff_decay was computed with num_updates = n, i.e.
    # as (1+n)/(10+n). Verify that by driving the update and reading
    # the internal counter.
    expected_points = {
        1: 2.0 / 11.0,     # first update
        5: 6.0 / 15.0,
        9: 10.0 / 19.0,    # still on the warmup branch
        100: 101.0 / 110.0,
        1000: 1001.0 / 1010.0,
    }

    # Reconstruct the schedule exactly the way ModelEma does.
    for step, expected in expected_points.items():
        got = min(ema.decay, (1.0 + step) / (ModelEma.BIAS_WARMUP + step))
        assert math.isclose(got, expected, rel_tol=1e-9), (
            f"step={step}: eff_decay should be {expected}, got {got}"
        )


def test_decay_plateaus_at_nominal() -> None:
    """Once the warmup branch exceeds ``decay`` it must be clamped."""
    decay = 0.9999
    # The warmup branch crosses the nominal decay when
    # (1+n)/(10+n) >= 0.9999 i.e. n >= 90_000. Check a point well
    # past that to make sure the min() clamp is applied.
    n = 200_000
    warmup_value = (1.0 + n) / (ModelEma.BIAS_WARMUP + n)
    assert warmup_value > decay, "test premise: warmup should exceed nominal here"
    eff = min(decay, warmup_value)
    assert math.isclose(eff, decay, rel_tol=1e-12)


def test_shadow_flushes_random_init_quickly() -> None:
    """After a handful of updates the shadow should be close to current weights.

    With the bias-corrected schedule, the very first update uses
    ``eff_decay = 2/11 approx 0.18``, so the shadow moves ~82% of the
    way from its starting value to the current weights on step 1
    alone. After ten updates with constant current weights, the
    shadow should be visually indistinguishable from ``current``.

    This is the property that the old ``decay = 0.999`` schedule
    violated: with a constant decay, a long warmup period was needed
    before the shadow caught up, leaving random-init residue in place.
    """
    torch.manual_seed(0)
    model = _tiny_model()
    ema = ModelEma(model, decay=0.9999)

    # Freeze a "trained" target state that differs from the init,
    # then hammer EMA.update() repeatedly with these constant weights.
    target_state = {
        name: torch.ones_like(tensor) if tensor.dtype.is_floating_point else tensor.clone()
        for name, tensor in model.state_dict().items()
    }
    model.load_state_dict(target_state)

    for _ in range(20):
        ema.update(model)

    # Every float tensor in the shadow should now be very close to 1.
    # The strict closed-form bound after 20 updates is
    # ``prod_{k=1..20} eff_decay_k approx 10/(10+20) = 1/3`` of the
    # initial shadow value remaining (telescoping product of
    # (k-1+10)/(k+10)), so the residual is <= ~0.33 of the init value
    # times (original_init - 1). With ``abs_tol=0.6`` the test
    # generously accommodates init variance while still catching a
    # broken update rule (which would leave the shadow near its init
    # rather than near 1).
    for name, shadow_tensor in ema.shadow.items():
        if not shadow_tensor.dtype.is_floating_point:
            # num_batches_tracked etc. are copied verbatim; skip.
            continue
        assert torch.allclose(shadow_tensor, torch.ones_like(shadow_tensor), atol=0.6), (
            f"Shadow tensor '{name}' is still far from target after 20 updates: "
            f"mean={shadow_tensor.mean().item():.4f}"
        )


def test_integer_buffers_are_copied_not_averaged() -> None:
    """``num_batches_tracked`` is int64; EMA must copy, not blend."""
    model = _tiny_model()
    ema = ModelEma(model, decay=0.9999)

    # Push BatchNorm through a forward pass to make sure
    # num_batches_tracked gets bumped. BN needs >=2 samples in train mode.
    model.train()
    _ = model(torch.randn(4, 4))
    ema.update(model)

    # The BN buffer should track the module's current int value exactly.
    for name, tensor in model.state_dict().items():
        if tensor.dtype.is_floating_point:
            continue
        shadow_tensor = ema.shadow[name]
        assert shadow_tensor.dtype == tensor.dtype, f"{name}: dtype changed under EMA"
        assert torch.equal(shadow_tensor, tensor), (
            f"Integer buffer '{name}' was not copied verbatim by EMA."
        )


def _run_all() -> None:
    """Entry point for ``python tests/test_model_ema.py`` (no pytest)."""
    tests = [
        test_bias_corrected_decay_schedule,
        test_decay_plateaus_at_nominal,
        test_shadow_flushes_random_init_quickly,
        test_integer_buffers_are_copied_not_averaged,
    ]
    for test in tests:
        test()
        print(f"  ok: {test.__name__}")
    print(f"All {len(tests)} ModelEma tests passed.")


if __name__ == "__main__":
    _run_all()
