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

Defaults are 80 epochs for each Task 2 (custom CNN) run and 15 for
each Task 1 (transfer learning) run. The brief's "e.g. 10–30
epochs" note is soft guidance ("depending on dataset and
hardware"); the shorter 30-epoch budget left the from-scratch CNN
clearly underfit on this ~3k-image split (the cosine schedule had
already annealed to zero at epoch 30 while training accuracy was
still climbing through 37 percent), so the custom budget is raised
to 80. Transfer learning converges far faster on pretrained
features and stays at 15. Both values are exposed as CLI flags
(``--custom-epochs`` and ``--transfer-epochs``) for easy adjustment.

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
import re
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
    parser.add_argument("--custom-epochs", type=int, default=80,
                        help="Epochs for each Task 2 (custom CNN) run. The brief's "
                             "'e.g. 10-30 epochs' is soft guidance explicitly scoped to "
                             "dataset and hardware availability. On Oxford-IIIT Pet "
                             "(~3k training images, ~80 per class) a from-scratch CNN "
                             "with strong augmentation + Mixup needs substantially more "
                             "gradient steps to reach its asymptote: the previous 30-epoch "
                             "cosine run finished with train accuracy at only 37 percent, "
                             "and the scheduler had already annealed the learning rate to "
                             "zero. 80 epochs leaves the cosine curve with useful head-room "
                             "past the 60-epoch mark and still runs in about 4.5 minutes "
                             "per experiment on an RTX 5880 Ada (six experiments, ~30 min "
                             "total). See Section 2.2 of the brief for the soft-guidance "
                             "wording.")
    parser.add_argument("--transfer-epochs", type=int, default=15,
                        help="Epochs for each Task 1 (transfer learning) run. Pretrained "
                             "features converge in far fewer steps than Task 2.")
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

    A glob of ``<name>_*`` is not safe here: if one experiment name is a
    prefix of another (for example ``custom_aug`` prefixes
    ``custom_aug_sched``), that glob silently captures the longer one
    and aggregation picks the wrong row. Match the exact
    ``<name>_YYYYMMDD_HHMMSS`` pattern with a regex instead so
    ``custom_aug`` only matches directories starting with
    ``custom_aug_<digits>_<digits>``.
    """
    pattern = re.compile(rf"^{re.escape(experiment_name)}_\d{{8}}_\d{{6}}$")
    matches = sorted(
        entry for entry in output_dir.iterdir()
        if entry.is_dir() and pattern.match(entry.name)
    )
    return matches[-1] if matches else None


VIS_MODULE = "pet_cw.visualize"

# Which experiment names should get Grad-CAM / prediction grids written
# at the end of the sweep. Task 2 best is ``custom_full``, Task 1 best is
# the fine-tuned ResNet-18. Frozen ResNet-18 is skipped on purpose: it
# shares the same backbone weights as the fine-tuned version minus the
# head, so its Grad-CAM figures would not add new information to the
# report.
VISUALISATION_TARGETS = (
    "custom_full",
    "transfer_resnet18_finetune",
)

# Each target produces one grid per (split, filter) combination so the
# report can show both the model's successes and its characteristic
# failure modes on the held-out test set.
VISUALISATION_VIEWS = (
    ("val", "incorrect"),
    ("val", "correct"),
    ("test", "incorrect"),
    ("test", "correct"),
)


def run_visualisations(
    output_dir: Path,
    experiments: list[dict],
    env: dict[str, str],
) -> None:
    """Render prediction + Grad-CAM grids for the two headline models.

    Called once after the whole sweep finishes so the figures show the
    final trained checkpoints. Failures are logged but do not abort the
    sweep: the training results remain valid even if one matplotlib
    call goes wrong.
    """
    ran_names = {experiment["name"] for experiment in experiments}
    for target in VISUALISATION_TARGETS:
        if target not in ran_names:
            continue
        run_dir = find_latest_run_dir(output_dir, target)
        if run_dir is None:
            print(f"  [visualise] No run folder found for {target}, skipping.")
            continue
        checkpoint = run_dir / "best_model.pt"
        if not checkpoint.exists():
            print(f"  [visualise] {checkpoint} missing, skipping.")
            continue

        for split, filter_mode in VISUALISATION_VIEWS:
            print(f"  [visualise] {target} / {split} / {filter_mode}")
            command = [
                sys.executable, "-m", VIS_MODULE,
                "--checkpoint", str(checkpoint),
                "--split", split,
                "--filter", filter_mode,
                "--num-samples", "16",
                "--ncols", "4",
            ]
            result = subprocess.run(command, check=False, env=env)
            if result.returncode != 0:
                print(
                    f"  [visualise] Grid failed for {target} "
                    f"({split}/{filter_mode}); continuing with the rest."
                )


def plot_ablation_chart(rows: list[dict], output_dir: Path) -> Path | None:
    """Render a grouped bar chart of val / test accuracy across the sweep.

    Importing matplotlib is deferred to this function so a broken
    matplotlib install cannot prevent the training sweep from running.
    Returns the path to the PNG, or ``None`` if nothing could be plotted.
    """
    if not rows:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"  [chart] matplotlib unavailable ({exc}); skipping ablation_chart.png.")
        return None

    # Preserve the order experiments were defined in so the chart reads
    # A → F → Task 1, matching the report's narrative.
    ordered = [row for row in rows if "experiment_name" in row]
    if not ordered:
        return None

    names = [row["experiment_name"] for row in ordered]
    val_accs = [float(row.get("best_val_acc") or 0.0) * 100.0 for row in ordered]
    test_accs = [float(row.get("test_acc") or 0.0) * 100.0 for row in ordered]

    import numpy as np  # local import to match the matplotlib pattern above

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * len(names)), 5.0))
    bars_val = ax.bar(x - width / 2, val_accs, width, label="Validation accuracy", color="#3b7dd8")
    bars_test = ax.bar(x + width / 2, test_accs, width, label="Test accuracy", color="#d86b3b")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("COMP3065 ablation: validation vs held-out test accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    # Annotate each bar with its numeric value to save the report writer
    # having to squint at the axis.
    for bar_group in (bars_val, bars_test):
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    fig.tight_layout()
    chart_path = output_dir / "ablation_chart.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return chart_path


def aggregate_summaries(
    output_dir: Path,
    experiments: list[dict],
) -> tuple[Path | None, list[dict]]:
    """Collect ``summary.json`` from every completed run into one CSV + JSON.

    Returns ``(csv_path, rows)`` where ``csv_path`` is ``None`` if no
    run produced a summary. The raw row list is returned too so the
    caller can feed the same data into the ablation bar chart without
    re-reading the JSON.
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
        return None, []

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

    return csv_path, rows


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

    aggregate_path, aggregate_rows = aggregate_summaries(output_dir, experiments)
    chart_path = plot_ablation_chart(aggregate_rows, output_dir)

    # Grad-CAM + prediction grids for the two headline models. Only run
    # them if at least one of them was actually trained in this invocation
    # (``--only`` lets the user restrict the sweep to a subset).
    print()
    print("=" * 78)
    print("  Rendering visualisations for the headline models")
    print("=" * 78)
    run_visualisations(output_dir, experiments, env)

    print()
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
    if chart_path is not None:
        print(f"  Ablation bar chart:   {chart_path}")

    # Non-zero exit if anything failed, so CI / shell chaining can detect it.
    if any(code != 0 for _, code, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
