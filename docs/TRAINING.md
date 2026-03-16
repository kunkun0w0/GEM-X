# Training & Evaluation

## Dataset

GEM is trained on **Bones RigPlay-1**, an internal NVIDIA synthetic dataset. Bones RigPlay-1 is **not publicly released**.

Expected directory layout:

```
inputs/
└── metrosim_data_A2G_Bones2_DH/
    ├── train/
    ├── val/
    └── mocap/
```

Update `data_root` in `configs/train_datasets/metrosim_dh_train.yaml` if your data is located elsewhere.

## Training

### Single-GPU

```bash
python scripts/train.py exp=gem_soma_regression
```

### Multi-GPU (DDP)

```bash
python scripts/train.py exp=gem_soma_regression pl_trainer.devices=4
```

### Key Config Settings

The main experiment config is `configs/exp/gem_soma_regression.yaml`:

| Setting | Value |
|---|---|
| Body model | SOMA |
| Max steps | 500,000 |
| Precision | 16-mixed |
| Optimizer | AdamW (lr=2e-4) |
| Gradient clipping | 0.5 |
| Validation interval | Every 3,000 steps |

### W&B Logging

Logging uses Weights & Biases by default. To disable:

```bash
python scripts/train.py exp=gem_soma_regression use_wandb=false
```

## Evaluation

```bash
python scripts/train.py exp=gem_soma_regression task=test
```

This runs evaluation on the MetroSim validation split and reports per-frame SOMA body pose and global translation accuracy metrics.

## Hydra Config System

GEM uses [Hydra](https://hydra.cc/) for configuration management. Key config groups:

| Group | Description |
|---|---|
| `exp/` | Experiment configs (e.g., `gem_soma_regression`) |
| `model/` | Model architecture |
| `network/` | Network details (denoiser, regression) |
| `pipeline/` | Training pipeline (loss weights, features) |
| `train_datasets/` | Training data configs |
| `test_datasets/` | Evaluation data configs |

Override any config value from the command line:

```bash
python scripts/train.py exp=gem_soma_regression pl_trainer.max_steps=100000 optimizer.lr=1e-4
```
