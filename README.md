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

## 2. Run all recommended experiments

A single Python driver runs the full ablation and both transfer
strategies. Expect about 90 minutes on a single modern GPU (RTX 3090
/ A5000 territory), longer on older cards.

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
    --only custom_baseline custom_aug_only

python scripts/run_recommended_experiments.py --dry-run
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
| `summary.json`                           | Best val accuracy, test accuracy, best model path     |

Test-set artifacts only appear if the run used `--test-at-end`,
which the recommended driver does for every experiment.

## 4. Individual commands

### Custom CNN baseline

```bash
python -m pet_cw.train \
  --experiment-name custom_baseline \
  --model custom \
  --image-size 160 \
  --batch-size 64 \
  --epochs 30 \
  --augmentation none \
  --weight-decay 0.0 \
  --dropout 0.30 \
  --scheduler none \
  --label-smoothing 0.0 \
  --download --amp --test-at-end
```

### Custom CNN + augmentation only (single-variable improvement)

```bash
python -m pet_cw.train \
  --experiment-name custom_aug_only \
  --model custom \
  --image-size 160 \
  --batch-size 64 \
  --epochs 30 \
  --augmentation strong \
  --weight-decay 0.0 \
  --dropout 0.30 \
  --scheduler none \
  --label-smoothing 0.0 \
  --download --amp --test-at-end
```

The difference between these two runs isolates the effect of data
augmentation. That is the mandatory experimental improvement
specified by the coursework brief.

### Custom CNN with every regulariser stacked

```bash
python -m pet_cw.train \
  --experiment-name custom_full_improvement \
  --model custom \
  --image-size 160 \
  --batch-size 64 \
  --epochs 30 \
  --augmentation strong \
  --weight-decay 1e-4 \
  --dropout 0.45 \
  --scheduler cosine \
  --label-smoothing 0.1 \
  --grad-clip 1.0 \
  --download --amp --test-at-end
```

### Transfer learning: frozen backbone (feature extraction)

```bash
python -m pet_cw.train \
  --experiment-name transfer_resnet18_frozen \
  --model resnet18 --pretrained --freeze-backbone \
  --image-size 224 --batch-size 32 --epochs 15 \
  --augmentation basic \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 \
  --scheduler cosine --label-smoothing 0.1 \
  --download --amp --test-at-end
```

### Transfer learning: full fine-tune with differential LR

```bash
python -m pet_cw.train \
  --experiment-name transfer_resnet18_finetune \
  --model resnet18 --pretrained \
  --image-size 224 --batch-size 32 --epochs 15 \
  --augmentation basic \
  --optimizer adamw --lr 1e-4 --head-lr-mult 10 \
  --weight-decay 1e-4 --scheduler cosine --label-smoothing 0.1 \
  --download --amp --test-at-end
```

`--lr 1e-4` is the backbone learning rate. `--head-lr-mult 10`
gives the fresh classifier head an effective `1e-3`, which is the
classical "fine-tune the old features slowly, train the new head
faster" recipe.

## 5. Re-score an existing checkpoint

```bash
python -m pet_cw.evaluate \
  --checkpoint outputs/custom_full_improvement_*/best_model.pt \
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

* Final reported numbers come from the **test split** (the 3669
  images in `split="test"`). The validation slice taken out of
  `trainval` is only used for model selection, never for a reported
  headline number. That is why every run ends with `--test-at-end`.
* The five runs in `run_recommended_experiments.py` form a single
  coherent story. Baseline vs `aug_only` is the clean
  single-variable ablation the coursework requires. `aug_only`
  vs `full_improvement` quantifies the marginal benefit of
  stacking scheduler + weight decay + label smoothing + higher
  dropout on top of augmentation. `frozen` vs `finetune` answers
  the Lab 6 question about which transfer strategy is better here.
* `--seed 42` by default. Pass `--deterministic` for bit-exact
  reruns, at a small throughput cost.
* `PetResNet` has about 2.7M parameters. The classifier head is a
  single `Linear(256, 37)` because global average pooling removes
  the need for a wide fully-connected block.
