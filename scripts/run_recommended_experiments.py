"""Run every experiment reported in the COMP3065 coursework submission.

The sweep has two parts:

1. Task 2 (Custom CNN) — a progressive, single-variable-at-a-time
   ablation starting from a pure baseline and adding one regularisation
   technique at each subsequent step:

   A. ``custom_baseline``
      Reference point. No augmentation, no LR schedule, no weight
      decay, no label smoothing, no Mixup. Everything else in the
      sweep measures its delta from this row.

   B. ``custom_aug``
      Adds strong data augmentation (RandomResizedCrop + flip +
      TrivialAugmentWide + RandomErasing) and nothing else. This is
      the single-variable experimental improvement the brief asks
      for in §4: exactly one setting changes relative to A, so the
      accuracy delta is attributable to augmentation alone.

   C. ``custom_aug_sched``
      Adds warmup + cosine LR annealing on top of B.

   D. ``custom_aug_sched_wd``
      Adds L2 weight decay (1e-4) on top of C.

   E. ``custom_aug_sched_wd_ls``
      Adds label smoothing 0.1 on top of D.

   F. ``custom_full``
      Adds Mixup (alpha = 0.2) on top of E. Together, A→F forms a
      clean story: each row isolates one technique's contribution so
      the report can attribute every accuracy gain to a specific
      design choice.

2. Task 1 (Transfer Learning) — two runs that answer the Lab 6
   freeze-vs-fine-tune question on a pretrained ResNet-18:

   G. ``transfer_resnet18_frozen``
      Feature-extraction strategy. Backbone frozen, only the 37-way
      classifier head trains. Cheap; gives a lower bound on what
      pretrained features already know about pets.

   H. ``transfer_resnet18_finetune``
      Full fine-tuning with a differential learning rate: head LR is
      10x the backbone LR. Typically the strongest configuration
      once enough labelled data is available.

Every run is 30 epochs for custom and 15 for transfer, which sits
inside the brief's 10–30 epoch guideline and keeps the whole sweep
comfortably under two hours on a single modern GPU.

Usage
-----

Defaults (data downloads to ``./data``, results written to ``./outputs``)::

    python scripts/run_recommended_experiments.py

Override locations, worker count, or run a subset::

    python scripts/run_recommended_experiments.py \\
        --data-dir /mnt/data --output-dir /mnt/out --num-workers 8

    python scripts/run_recommended_experiments.py --only custom_baseline custom_aug

    python scripts/run_recommended_experiments.py --dry-run

A plain Python driver is used (not a shell script) because the
coursework brief specifies Python scripts or notebooks as the code
deliverable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
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
OXFORD_PET_HOST = "www.robots.ox.ac.uk"


OFFLINE_DATA_HELP = """Oxford-IIIT Pet is not present locally and this machine cannot resolve the dataset host.

Prepare the dataset manually on a machine with internet, then copy it to this server:

  1. Download:
       https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
       https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

  2. On the Linux GPU server, put both archives under:
       {base_folder}

  3. Extract them there:
       mkdir -p {base_folder}
       tar -xzf images.tar.gz -C {base_folder}
       tar -xzf annotations.tar.gz -C {base_folder}

  4. Expected final structure:
       {base_folder}/images/*.jpg
       {base_folder}/annotations/trainval.txt
       {base_folder}/annotations/test.txt

After that, rerun this script. The torchvision loader will see the files and skip downloading.
"""


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


def oxford_pet_ready(data_dir: str) -> bool:
    base_folder = Path(data_dir) / "oxford-iiit-pet"
    required = [
        base_folder / "images",
        base_folder / "annotations",
        base_folder / "annotations" / "trainval.txt",
        base_folder / "annotations" / "test.txt",
    ]
    return all(path.exists() for path in required)


def can_resolve_dataset_host() -> bool:
    try:
        socket.getaddrinfo(OXFORD_PET_HOST, 443, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return True


def preflight_dataset(data_dir: str) -> None:
    if oxford_pet_ready(data_dir):
        print(f"Dataset found under: {Path(data_dir) / 'oxford-iiit-pet'}")
        return

    if can_resolve_dataset_host():
        print("Dataset not found locally; torchvision will download Oxford-IIIT Pet on the first run.")
        return

    base_folder = Path(data_dir) / "oxford-iiit-pet"
    raise SystemExit(OFFLINE_DATA_HELP.format(base_folder=base_folder))


def build_experiments(
    data_dir: str,
    output_dir: str,
    num_workers: int,
    custom_epochs: int,
    transfer_epochs: int,
) -> list[dict]:
    """Return one dict per experiment in the ablation.

    Each dict is {"name": str, "args": list[str]}. Args are passed
    verbatim to ``python -m pet_cw.train``. Experiments are split
    into per-run dicts so a filter (``--only``) can select any
    subset without juggling shared-state variables.
    """
    # Training settings held constant across every custom run in the
    # progressive ablation. Only the variable under test changes.
    common_custom = [
        "--model", "custom",
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--image-size", "224",
        "--batch-size", "64",
        "--epochs", str(custom_epochs),
        "--optimizer", "adamw",
        "--lr", "1e-3",
        "--dropout", "0.3",
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
        "--epochs", str(transfer_epochs),
        "--augmentation", "basic",
        "--optimizer", "adamw",
        "--weight-decay", "1e-4",
        "--scheduler", "cosine",
        "--warmup-epochs", "1",
        "--label-smoothing", "0.1",
        "--num-workers", str(num_workers),
        "--download",
        "--amp",
        "--test-at-end",
    ]

    return [
        # ---- Task 2: progressive single-variable ablation ----
        {
            "name": "custom_baseline",
            "args": [
                "--experiment-name", "custom_baseline",
                *common_custom,
                "--augmentation", "none",
                "--scheduler", "none",
                "--weight-decay", "0.0",
                "--label-smoothing", "0.0",
                "--mixup-alpha", "0.0",
            ],
        },
        {
            "name": "custom_aug",
            "args": [
                "--experiment-name", "custom_aug",
                *common_custom,
                "--augmentation", "strong",
                "--scheduler", "none",
                "--weight-decay", "0.0",
                "--label-smoothing", "0.0",
                "--mixup-alpha", "0.0",
            ],
        },
        {
            "name": "custom_aug_sched",
            "args": [
                "--experiment-name", "custom_aug_sched",
                *common_custom,
                "--augmentation", "strong",
                "--scheduler", "cosine",
                "--warmup-epochs", "3",
                "--weight-decay", "0.0",
                "--label-smoothing", "0.0",
                "--mixup-alpha", "0.0",
            ],
        },
        {
            "name": "custom_aug_sched_wd",
            "args": [
                "--experiment-name", "custom_aug_sched_wd",
                *common_custom,
                "--augmentation", "strong",
                "--scheduler", "cosine",
                "--warmup-epochs", "3",
                "--weight-decay", "1e-4",
                "--label-smoothing", "0.0",
                "--mixup-alpha", "0.0",
            ],
        },
        {
            "name": "custom_aug_sched_wd_ls",
            "args": [
                "--experiment-name", "custom_aug_sched_wd_ls",
                *common_custom,
                "--augmentation", "strong",
                "--scheduler", "cosine",
                "--warmup-epochs", "3",
                "--weight-decay", "1e-4",
                "--label-smoothing", "0.1",
                "--mixup-alpha", "0.0",
            ],
        },
        {
            "name": "custom_full",
            "args": [
                "--experiment-name", "custom_full",
                *common_custom,
                "--augmentation", "strong",
                "--scheduler", "cosine",
                "--warmup-epochs", "3",
                "--weight-decay", "1e-4",
                "--label-smoothing", "0.1",
                "--mixup-alpha", "0.2",
            ],
        },
        # ---- Task 1: freeze vs fine-tune on ResNet-18 ----
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
    parser.add_argument("--custom-epochs", type=int, default=30,
                        help="Epochs for each Task 2 (custom CNN) run. Brief allows 10-30.")
    parser.add_argument("--transfer-epochs", type=int, default=15,
                        help="Epochs for each Task 1 (transfer learning) run.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Subset of experiment names to run (default: all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands that would be executed and exit.")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort the whole sweep if any single run exits non-zero.")
    parser.add_argument("--skip-dataset-check", action="store_true",
                        help="Skip the local / DNS preflight check for the Oxford-IIIT Pet dataset.")
    return parser.parse_args()


def format_command(args: list[str]) -> str:
    """Pretty-print a command for logging."""
    return " ".join([sys.executable, "-m", TRAIN_MODULE, *args])


def run_single_experiment(args: list[str], env: dict[str, str]) -> int:
    """Launch one training run. Return its exit code."""
    command = [sys.executable, "-m", TRAIN_MODULE, *args]
    # Inherit stdout/stderr so tqdm bars and log lines stream live.
    process = subprocess.run(command, check=False, env=env)
    return process.returncode


def find_latest_run_dir(output_dir: Path, experiment_name: str) -> Path | None:
    """Return the most recent output folder produced by ``experiment_name``.

    Runs are written as ``<name>_<timestamp>`` so sorting by name finds
    the latest deterministically. Returns ``None`` if no matching folder
    exists.
    """
    matches = sorted(output_dir.glob(f"{experiment_name}_*"))
    return matches[-1] if matches else None


def aggregate_summaries(output_dir: Path, experiments: list[dict]) -> Path | None:
    """Collect ``summary.json`` from every completed run into one CSV + JSON.

    Returns the CSV path, or ``None`` if no run produced a summary.
    This saves having to re-open individual history files when the
    report draws its comparison table.
    """
    rows: list[dict] = []
    for experiment in experiments:
        run_dir = find_latest_run_dir(output_dir, experiment["name"])
        if run_dir is None:
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["run_dir"] = str(run_dir)
        rows.append(summary)

    if not rows:
        return None

    # Use the union of keys so a column does not get silently dropped
    # just because one run is missing it (e.g. test_acc without --test-at-end).
    all_keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    csv_path = output_dir / "ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in all_keys})

    json_path = output_dir / "ablation_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    return csv_path


def main() -> None:
    cli = parse_args()

    data_dir = project_path(cli.data_dir)
    output_dir = Path(project_path(cli.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(
        data_dir=data_dir,
        output_dir=str(output_dir),
        num_workers=cli.num_workers,
        custom_epochs=cli.custom_epochs,
        transfer_epochs=cli.transfer_epochs,
    )

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

    if not cli.skip_dataset_check and not cli.dry_run:
        preflight_dataset(data_dir)

    print(f"Planning to run {len(experiments)} experiment(s):")
    for experiment in experiments:
        print(f"  - {experiment['name']}")
    print()

    if cli.dry_run:
        for experiment in experiments:
            print(f"[dry-run] {experiment['name']}")
            print(f"    {format_command(experiment['args'])}")
        return

    env = subprocess_env()
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
        exit_code = run_single_experiment(experiment["args"], env)
        elapsed = time.time() - run_start
        results.append((name, exit_code, elapsed))

        print()
        print(f"  Finished {name}: exit_code={exit_code}, elapsed={elapsed/60:.1f} min")
        print()

        if exit_code != 0 and cli.stop_on_error:
            print(f"  Aborting sweep because {name} failed and --stop-on-error is set.")
            break

    total_elapsed = time.time() - sweep_start

    aggregate_path = aggregate_summaries(output_dir, experiments)

    print("=" * 78)
    print("  Sweep summary")
    print("=" * 78)
    for name, exit_code, elapsed in results:
        status = "OK" if exit_code == 0 else f"FAILED ({exit_code})"
        print(f"  {name:<32s} {status:<14s} {elapsed/60:6.1f} min")
    print(f"  Total wall-clock: {total_elapsed/60:.1f} min")
    print(f"  Result folders under: {output_dir}")
    if aggregate_path is not None:
        print(f"  Aggregated table:     {aggregate_path}")

    # Non-zero exit if anything failed, so CI / shell chaining can detect it.
    if any(code != 0 for _, code, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
