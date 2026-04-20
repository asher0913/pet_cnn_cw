"""Run every experiment reported in the COMP3065 coursework submission.

Design of the study
-------------------

A. ``custom_baseline``
   Task 2 baseline. No augmentation, no scheduler, no weight decay,
   no label smoothing. Acts as the reference point; every later
   improvement is measured as a delta from this row.

B. ``custom_aug_only``
   Same as A, but with strong data augmentation turned on. Exactly
   one variable changes compared to the baseline, so the resulting
   accuracy delta is attributable to augmentation alone. This is
   the single-variable experimental improvement the brief asks for.

C. ``custom_full_improvement``
   Strong augmentation + cosine LR schedule + L2 weight decay +
   label smoothing + higher dropout + gradient clipping. Not a
   single-variable ablation on top of B; instead a "stack every
   regulariser" run that shows how much further the Task 2 model
   can be pushed when the improvements compound.

D. ``transfer_resnet18_frozen``
   Task 1, feature-extraction strategy. Pretrained ResNet-18 with
   the backbone frozen, so only the 37-way classifier head trains.
   The cheaper of the two transfer-learning strategies.

E. ``transfer_resnet18_finetune``
   Task 1, full fine-tuning strategy. Same network, but the
   backbone is trainable with a differential learning rate (head
   learning rate ten times the backbone learning rate). The
   strategy that usually wins once enough labelled data is
   available.

Comparisons
-----------

* A vs B         : clean single-variable ablation (augmentation).
* B vs C         : diminishing-returns argument for stacked regularisers.
* D vs E         : Lab 6 question on freeze-vs-fine-tune.
* C vs {D, E}    : custom CNN against a pretrained alternative.

Usage
-----

Defaults (data downloads to ``./data``, results written to ``./outputs``)::

    python scripts/run_recommended_experiments.py

Override locations, worker count, or a subset of experiments::

    python scripts/run_recommended_experiments.py \
        --data-dir /mnt/data --output-dir /mnt/out --num-workers 8

    python scripts/run_recommended_experiments.py --only custom_baseline custom_aug_only

A plain Python driver is used (rather than a shell script) because
the coursework brief specifies Python scripts or notebooks as the
code deliverable.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


# Module-path form is used so the driver works whether executed from
# the repo root or from anywhere else, as long as ``pet_cw`` is
# importable in the current environment.
TRAIN_MODULE = "pet_cw.train"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def project_path(path: str) -> str:
    """Resolve relative paths against the project root, not the caller's cwd."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def subprocess_env() -> dict[str, str]:
    """Make the src-layout package importable even before pip install -e ."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_DIR) if not existing else f"{SRC_DIR}{os.pathsep}{existing}"
    return env


def build_experiments(data_dir: str, output_dir: str, num_workers: int) -> list[dict]:
    """Return one dict per experiment in the ablation.

    Each dict is {"name": str, "args": list[str]}. Args are passed
    verbatim to ``python -m pet_cw.train``. Experiments are split
    into per-run dicts so a filter (``--only``) can select any
    subset without juggling shared-state variables.
    """
    common_custom = [
        "--model", "custom",
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--image-size", "160",
        "--batch-size", "64",
        "--epochs", "30",
        "--optimizer", "adamw",
        "--lr", "1e-3",
        "--num-workers", str(num_workers),
        "--download",
        "--amp",
        "--test-at-end",
    ]

    common_transfer = [
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--image-size", "224",
        "--batch-size", "32",
        "--epochs", "15",
        "--augmentation", "basic",
        "--optimizer", "adamw",
        "--weight-decay", "1e-4",
        "--scheduler", "cosine",
        "--label-smoothing", "0.1",
        "--num-workers", str(num_workers),
        "--download",
        "--amp",
        "--test-at-end",
    ]

    return [
        {
            "name": "custom_baseline",
            "args": [
                "--experiment-name", "custom_baseline",
                *common_custom,
                "--augmentation", "none",
                "--weight-decay", "0.0",
                "--dropout", "0.3",
                "--scheduler", "none",
                "--label-smoothing", "0.0",
            ],
        },
        {
            "name": "custom_aug_only",
            "args": [
                "--experiment-name", "custom_aug_only",
                *common_custom,
                "--augmentation", "strong",
                "--weight-decay", "0.0",
                "--dropout", "0.3",
                "--scheduler", "none",
                "--label-smoothing", "0.0",
            ],
        },
        {
            "name": "custom_full_improvement",
            "args": [
                "--experiment-name", "custom_full_improvement",
                *common_custom,
                "--augmentation", "strong",
                "--weight-decay", "1e-4",
                "--dropout", "0.45",
                "--scheduler", "cosine",
                "--label-smoothing", "0.1",
                "--grad-clip", "1.0",
            ],
        },
        {
            "name": "transfer_resnet18_frozen",
            "args": [
                "--experiment-name", "transfer_resnet18_frozen",
                "--model", "resnet18",
                "--pretrained",
                "--freeze-backbone",
                *common_transfer,
                "--lr", "1e-3",
            ],
        },
        {
            "name": "transfer_resnet18_finetune",
            "args": [
                "--experiment-name", "transfer_resnet18_finetune",
                "--model", "resnet18",
                "--pretrained",
                *common_transfer,
                "--lr", "1e-4",
                "--head-lr-mult", "10",
            ],
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full COMP3065 ablation as a sequence of training jobs.",
    )
    parser.add_argument("--data-dir", default="./data",
                        help="Where the Oxford-IIIT Pet dataset lives (or will be downloaded to).")
    parser.add_argument("--output-dir", default="./outputs",
                        help="Parent directory for per-experiment result folders.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes per run.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Subset of experiment names to run (default: all five).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands that would be executed and exit.")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort the whole sweep if any single run exits non-zero.")
    return parser.parse_args()


def format_command(args: list[str]) -> str:
    """Pretty-print a command for logging."""
    command = [sys.executable, "-m", TRAIN_MODULE, *args]
    command_text = " ".join(shlex.quote(part) for part in command)
    return f"cd {shlex.quote(str(PROJECT_ROOT))} && PYTHONPATH={shlex.quote(str(SRC_DIR))} {command_text}"


def run_single_experiment(args: list[str]) -> int:
    """Launch one training run. Return its exit code."""
    command = [sys.executable, "-m", TRAIN_MODULE, *args]
    # Inherit stdout/stderr so tqdm bars and log lines stream live.
    process = subprocess.run(command, check=False, cwd=PROJECT_ROOT, env=subprocess_env())
    return process.returncode


def main() -> None:
    cli = parse_args()
    cli.data_dir = project_path(cli.data_dir)
    cli.output_dir = project_path(cli.output_dir)

    Path(cli.output_dir).mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(cli.data_dir, cli.output_dir, cli.num_workers)

    if cli.only:
        wanted = set(cli.only)
        known = {experiment["name"] for experiment in experiments}
        unknown = wanted - known
        if unknown:
            raise SystemExit(
                f"Unknown experiment name(s): {sorted(unknown)}. "
                f"Choose from: {sorted(known)}"
            )
        experiments = [experiment for experiment in experiments if experiment["name"] in wanted]

    print(f"Planning to run {len(experiments)} experiment(s):")
    for experiment in experiments:
        print(f"  - {experiment['name']}")
    print()

    if cli.dry_run:
        for experiment in experiments:
            print(f"[dry-run] {experiment['name']}")
            print(f"    {format_command(experiment['args'])}")
        return

    results: list[tuple[str, int, float]] = []
    sweep_start = time.time()

    for experiment in experiments:
        name = experiment["name"]
        print("=" * 78)
        print(f"  Starting experiment: {name}")
        print("=" * 78)
        print(format_command(experiment["args"]))
        print()

        run_start = time.time()
        exit_code = run_single_experiment(experiment["args"])
        elapsed = time.time() - run_start
        results.append((name, exit_code, elapsed))

        print()
        print(f"  Finished {name}: exit_code={exit_code}, elapsed={elapsed/60:.1f} min")
        print()

        if exit_code != 0 and cli.stop_on_error:
            print(f"  Aborting sweep because {name} failed and --stop-on-error is set.")
            break

    total_elapsed = time.time() - sweep_start
    print("=" * 78)
    print("  Sweep summary")
    print("=" * 78)
    for name, exit_code, elapsed in results:
        status = "OK" if exit_code == 0 else f"FAILED ({exit_code})"
        print(f"  {name:<32s} {status:<14s} {elapsed/60:6.1f} min")
    print(f"  Total wall-clock: {total_elapsed/60:.1f} min")
    print(f"  Result folders under: {cli.output_dir}")

    # Non-zero exit if anything failed, so CI / shell chaining can detect it.
    if any(code != 0 for _, code, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
