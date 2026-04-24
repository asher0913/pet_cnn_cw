# COMP3065 Coursework: Oxford-IIIT Pet Classification

Source code for the COMP3065 coursework. Two tasks are implemented
in the same code base:

* **Task 1 (Transfer Learning).** Pretrained torchvision networks
  (ResNet, VGG, MobileNet). Two strategies are supported: feature
  extraction with a frozen backbone, and full fine-tuning with
  differential learning rates.
* **Task 2 (Custom CNN).** `PetResNet`, a residual network with
  Squeeze-and-Excite channel attention designed specifically for
  this dataset. Trained from scratch, no pretrained weights.

The project also contains a proper single-variable ablation for the
mandatory experimental improvement, Grad-CAM visualisation, and
both validation and held-out test-set evaluation.

## Project layout

```
pet_cnn_cw/
├── README.md
├── pyproject.toml
├── requirements.txt
├── scripts/
│   └── run_recommended_experiments.py
└── src/pet_cw/
    ├── __init__.py
    ├── data.py          # dataset, train/val split, transforms, test loader
    ├── models.py        # PetResNet, transfer wrappers, differential LR helper
    ├── train.py         # CLI training entry point
    ├── evaluate.py      # re-score a checkpoint on val + test
    ├── predict.py       # top-k inference on files or a folder
    ├── visualize.py     # prediction grids + Grad-CAM grids
    ├── gradcam.py       # from-scratch Grad-CAM implementation
    └── utils.py         # seeding, metrics, plotting, checkpoint helpers
```

## 1. Linux + NVIDIA GPU setup

From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-compatible PyTorch build that matches the target
machine. The command below works on CUDA 12.1; substitute the right
index URL from https://pytorch.org/get-started/locally/ for other
CUDA versions.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

Quick sanity check:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

If `torch.cuda.is_available()` is `False`, reinstall `torch` with
the CUDA build that matches the installed driver.

## 1.1 Dataset setup on servers without internet

The recommended script passes `--download`, so torchvision will try
to fetch Oxford-IIIT Pet automatically. If the GPU server has no
internet access or DNS, the training process will raise an error
of the form:

```text
Temporary failure in name resolution
urllib.error.URLError
```

That is a server network problem, not a training-code problem. In
that case, download the two archives on another machine and copy
them to the server:

```text
https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```

Then prepare the expected torchvision folder:

```bash
cd /home/unnc/zhang/pet_cnn_cw
mkdir -p data/oxford-iiit-pet

# Put images.tar.gz and annotations.tar.gz in data/oxford-iiit-pet first.
tar -xzf data/oxford-iiit-pet/images.tar.gz -C data/oxford-iiit-pet
tar -xzf data/oxford-iiit-pet/annotations.tar.gz -C data/oxford-iiit-pet
```

Expected final structure:

```text
data/oxford-iiit-pet/images/*.jpg
data/oxford-iiit-pet/annotations/trainval.txt
data/oxford-iiit-pet/annotations/test.txt
```

Once those files exist, rerun:

```bash
python scripts/run_recommended_experiments.py
```

## 2. Run all recommended experiments

A single Python driver runs the full Task 2 ablation (seven custom-CNN
runs, 80 epochs each) plus all three Task 1 transfer strategies
(frozen / fine-tune / fine-tune + EMA, 15 epochs each). Ten runs in
total. End-to-end wall-clock is about 35-40 minutes on an RTX 5880 Ada
/ RTX 4090, roughly 85 minutes on an RTX 3090 / A5000, longer on older
cards.

The last custom-CNN row (`custom_full_ema`) and the last transfer row
(`transfer_resnet18_finetune_ema`) layer three generic
stability / test-time techniques on top of the preceding row:

* **EMA** — an exponential moving average of model weights (Polyak
  averaging, standard in EfficientNet / ConvNeXt / DeiT). The shadow
  weights are used for validation and are persisted as
  `best_model.pt`.
* **`min_lr = 1e-5`** — non-zero floor on the cosine schedule so the
  last few epochs keep making small but useful updates.
* **Horizontal-flip TTA** — at the final `--test-at-end` evaluation,
  predictions are averaged over the image and its left-right flip.
  Training is untouched. `summary.json` records both the standard
  `test_acc` and the TTA `test_acc_tta` so the contribution of TTA
  can be attributed separately in the report.

None of EMA, TTA, or `min_lr` changes the architecture or adds
pretrained weights, so Task 2's "custom CNN, no pretrained weights"
constraint and Task 1's "permitted architectures" constraint are
both respected.

```bash
python scripts/run_recommended_experiments.py
```

Override locations, worker count, or run only part of the sweep:

```bash
python scripts/run_recommended_experiments.py \
    --data-dir /mnt/data \
    --output-dir /mnt/out \
    --num-workers 8

python scripts/run_recommended_experiments.py \
    --only custom_baseline custom_aug

python scripts/run_recommended_experiments.py --dry-run
```

Epoch budgets are configurable on the driver as well, should the
defaults need to be shortened for a smoke test or stretched for a
final report run:

```bash
python scripts/run_recommended_experiments.py \
    --custom-epochs 80 --transfer-epochs 15
```

The driver invokes `python -m pet_cw.train` for each experiment as a
subprocess, so each run uses the same code path as the standalone
commands in section 4.

## 3. What each run produces

A run named `foo` creates `outputs/foo_<timestamp>/` with:

| File                                     | What it is                                            |
| ---------------------------------------- | ----------------------------------------------------- |
| `best_model.pt`                          | Checkpoint with the highest validation accuracy       |
| `last_model.pt`                          | Final checkpoint after the last epoch                 |
| `run_config.json`                        | All CLI args plus class names and split sizes         |
| `history.csv`                            | Per-epoch loss, accuracy and learning rate            |
| `training_curves.png`                    | Loss and accuracy curves                              |
| `validation_confusion_matrix.png` / csv  | Best-epoch confusion matrix on the validation split   |
| `validation_classification_report.txt`   | Per-class precision / recall / F1 (validation)        |
| `validation_per_class_accuracy.png`      | Per-class accuracy bar chart (validation)             |
| `test_confusion_matrix.png` / csv        | Confusion matrix on the official test split          |
| `test_classification_report.txt`         | Per-class precision / recall / F1 (test)              |
| `test_per_class_accuracy.png`            | Per-class accuracy bar chart (test)                   |
| `test_tta_confusion_matrix.png` / csv    | TTA confusion matrix (only when `--tta` was used)     |
| `test_tta_classification_report.txt`     | Per-class report under TTA (only when `--tta` was used) |
| `test_tta_per_class_accuracy.png`        | Per-class accuracy bar chart under TTA                |
| `summary.json`                           | Best val accuracy, test accuracy (and TTA test accuracy), best model path |

Test-set artifacts only appear if the run used `--test-at-end`,
which the recommended driver does for every experiment.

Once every run in the sweep has finished, the driver also writes
three project-level artifacts plus a visualisation folder next to
each headline checkpoint:

| Artifact                                   | What it is                                           |
| ------------------------------------------ | ---------------------------------------------------- |
| `outputs/ablation_summary.csv`             | One row per experiment with every headline number   |
| `outputs/ablation_summary.json`            | Same information in JSON, easier to parse in notebooks |
| `outputs/ablation_chart.png`               | Grouped bar chart of val / test accuracy across the full sweep |
| `outputs/custom_full_ema_*/visualisation/`     | Prediction grids + Grad-CAM overlays for the Task 2 best model (val/test × correct/incorrect) |
| `outputs/transfer_resnet18_finetune_ema_*/visualisation/` | Same four grids for the Task 1 best model |

That is every figure and table the report needs: the confusion
matrices and training curves come from the individual run folders,
the headline comparison comes from `ablation_chart.png`, and the
Grad-CAM discussion draws on the `visualisation/` folders.

## 4. Individual commands

### Task 2 progressive ablation (Custom CNN)

The six runs below form a single-variable-at-a-time progression.
Every row adds exactly one technique compared with the row above,
so any accuracy delta can be attributed to that change alone.

```bash
# A. Baseline: nothing but the architecture.
python -m pet_cw.train \
  --experiment-name custom_baseline \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation none --scheduler none \
  --weight-decay 0.0 --label-smoothing 0.0 --mixup-alpha 0.0 \
  --download --amp --test-at-end

# B. + strong augmentation (this is the single-variable improvement
#    the coursework brief asks for in §4).
python -m pet_cw.train \
  --experiment-name custom_aug \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation strong --scheduler none \
  --weight-decay 0.0 --label-smoothing 0.0 --mixup-alpha 0.0 \
  --download --amp --test-at-end

# C. + warmup + cosine LR schedule.
python -m pet_cw.train \
  --experiment-name custom_aug_sched \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation strong --scheduler cosine --warmup-epochs 3 \
  --weight-decay 0.0 --label-smoothing 0.0 --mixup-alpha 0.0 \
  --download --amp --test-at-end

# D. + L2 weight decay.
python -m pet_cw.train \
  --experiment-name custom_aug_sched_wd \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation strong --scheduler cosine --warmup-epochs 3 \
  --weight-decay 1e-4 --label-smoothing 0.0 --mixup-alpha 0.0 \
  --download --amp --test-at-end

# E. + label smoothing.
python -m pet_cw.train \
  --experiment-name custom_aug_sched_wd_ls \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation strong --scheduler cosine --warmup-epochs 3 \
  --weight-decay 1e-4 --label-smoothing 0.1 --mixup-alpha 0.0 \
  --download --amp --test-at-end

# F. + Mixup.
python -m pet_cw.train \
  --experiment-name custom_full \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation strong --scheduler cosine --warmup-epochs 3 \
  --weight-decay 1e-4 --label-smoothing 0.1 --mixup-alpha 0.2 \
  --download --amp --test-at-end

# G. + EMA + cosine min_lr + TTA (the final, best Task 2 configuration).
#    EMA and min_lr change how training proceeds; TTA applies only at
#    --test-at-end and reports a second ``test_acc_tta`` number.
python -m pet_cw.train \
  --experiment-name custom_full_ema \
  --model custom --image-size 224 --batch-size 64 --epochs 80 \
  --optimizer adamw --lr 1e-3 --dropout 0.3 \
  --augmentation strong --scheduler cosine --warmup-epochs 3 \
  --weight-decay 1e-4 --label-smoothing 0.1 --mixup-alpha 0.2 \
  --ema --ema-decay 0.999 --min-lr 1e-5 --tta \
  --download --amp --test-at-end
```

Because `--pretrained` defaults to *off*, every Task 2 run here
trains from random initialisation even though the `custom` model
would ignore pretrained weights anyway; this keeps the saved
`run_config.json` honest for markers.

### Task 1 transfer learning (ResNet-18)

```bash
# Frozen backbone (feature extraction).
python -m pet_cw.train \
  --experiment-name transfer_resnet18_frozen \
  --model resnet18 --pretrained --freeze-backbone \
  --image-size 224 --batch-size 32 --epochs 15 \
  --augmentation basic \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 \
  --scheduler cosine --warmup-epochs 1 --label-smoothing 0.1 \
  --download --amp --test-at-end

# Full fine-tune with differential LR.
python -m pet_cw.train \
  --experiment-name transfer_resnet18_finetune \
  --model resnet18 --pretrained \
  --image-size 224 --batch-size 32 --epochs 15 \
  --augmentation basic \
  --optimizer adamw --lr 1e-4 --head-lr-mult 10 \
  --weight-decay 1e-4 --scheduler cosine --warmup-epochs 1 \
  --label-smoothing 0.1 \
  --download --amp --test-at-end

# Fine-tune + EMA + cosine min_lr + TTA (best Task 1 configuration).
python -m pet_cw.train \
  --experiment-name transfer_resnet18_finetune_ema \
  --model resnet18 --pretrained \
  --image-size 224 --batch-size 32 --epochs 15 \
  --augmentation basic \
  --optimizer adamw --lr 1e-4 --head-lr-mult 10 \
  --weight-decay 1e-4 --scheduler cosine --warmup-epochs 1 \
  --label-smoothing 0.1 \
  --ema --ema-decay 0.999 --min-lr 1e-5 --tta \
  --download --amp --test-at-end
```

`--lr 1e-4` is the backbone learning rate. `--head-lr-mult 10`
gives the fresh classifier head an effective `1e-3`, which is the
classical "fine-tune the old features slowly, train the new head
faster" recipe.

## 5. Re-score an existing checkpoint

```bash
python -m pet_cw.evaluate \
  --checkpoint outputs/custom_full_*/best_model.pt \
  --device auto
```

Produces fresh `validation_*` and `test_*` artifacts under an
`evaluation/` sub-folder beside the checkpoint.

## 6. Prediction and Grad-CAM visualisation

```bash
python -m pet_cw.visualize \
  --checkpoint outputs/transfer_resnet18_finetune_*/best_model.pt \
  --num-samples 16 --filter incorrect --split val
```

Writes `predictions_grid_val_incorrect.png` and
`gradcam_grid_val_incorrect.png` in a `visualisation/` folder next
to the checkpoint. Useful for the Discussion section: showing
Grad-CAM on the mistakes tends to reveal whether the model is
attending to the animal (confident but confused between two similar
breeds) or to the background (a genuinely bad feature-learning
problem).

Set `--filter correct` or `--filter any` for the other views.

## 7. Inference on new images

```bash
python -m pet_cw.predict \
  --checkpoint outputs/transfer_resnet18_finetune_*/best_model.pt \
  --input /path/to/image_or_folder \
  --output-csv predictions.csv \
  --top-k 5
```

## 8. Notes on methodology

* The validation accuracy is the headline number reported in the
  coursework write-up because the brief names "accuracy on the
  validation set" as the required evaluation metric. The held-out
  test split is only touched once per run at the very end (enabled
  by `--test-at-end`) so a single independent check of the same
  model is also available.
* The first six custom-CNN runs produced by
  `run_recommended_experiments.py` (A `custom_baseline` through F
  `custom_full`) form a single-variable-at-a-time progression so
  every increment from baseline to `custom_full` can be attributed
  to exactly one technique. `baseline → custom_aug` is the
  augmentation-only ablation the brief explicitly demands in §4;
  the remaining rows quantify the marginal contribution of LR
  scheduling, weight decay, label smoothing and Mixup. The seventh
  row, G `custom_full_ema`, layers EMA + cosine `min_lr` + TTA on
  top of F to measure how much additional headroom those generic
  training-stability / test-time techniques add once the
  algorithmic knobs are already tuned. On the transfer side,
  `transfer_resnet18_frozen` versus `transfer_resnet18_finetune`
  answers the Lab 6 question about feature extraction versus full
  fine-tuning; the third transfer row
  `transfer_resnet18_finetune_ema` adds the same EMA + `min_lr` +
  TTA package to fine-tuning, giving the strongest legal Task 1
  configuration.
* `--seed 42` by default. Pass `--deterministic` for bit-exact
  reruns, at a small throughput cost.
* `PetResNet` has about 2.7M parameters. The classifier head is a
  single `Linear(256, 37)` because global average pooling removes
  the need for a wide fully-connected block.
