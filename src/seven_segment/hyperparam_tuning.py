import time
from dataclasses import asdict

import optuna
import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from scripts.util.build_lmdb import LMDBScoreDataset
from src.util.gpu import get_device

from .data_augmentor import AugmentationConfig
from .train_model import (
    TrainTransform,
    ValTransform,
    build_model,
    make_weighted_sampler,
    run_epoch,
)


def objective(trial: optuna.Trial) -> float:
    cfg = AugmentationConfig(
        # Rotation
        rotation_max_deg=trial.suggest_float("rotation_max_deg", 3.0, 15.0),
        rotation_p=trial.suggest_float("rotation_p", 0.3, 0.8),
        # Perspective warp
        perspective_max_shift=trial.suggest_float("perspective_max_shift", 0.05, 0.20),
        perspective_p=trial.suggest_float("perspective_p", 0.3, 0.8),
        # Crop and pad
        crop_max_fraction=trial.suggest_float("crop_max_fraction", 0.02, 0.12),
        pad_max_fraction=trial.suggest_float("pad_max_fraction", 0.05, 0.15),
        crop_p=trial.suggest_float("crop_p", 0.3, 0.7),
        pad_p=trial.suggest_float("pad_p", 0.3, 0.7),
        # Brightness and contrast
        brightness_max_delta=trial.suggest_float("brightness_max_delta", 20.0, 80.0),
        contrast_range=(
            trial.suggest_float("contrast_low", 0.4, 0.9),
            trial.suggest_float("contrast_high", 1.1, 2.0),
        ),
        brightness_contrast_p=trial.suggest_float("brightness_contrast_p", 0.3, 0.8),
        # Gamma
        gamma_range=(
            trial.suggest_float("gamma_low", 0.4, 0.9),
            trial.suggest_float("gamma_high", 1.1, 2.0),
        ),
        gamma_p=trial.suggest_float("gamma_p", 0.2, 0.6),
        # Blur
        blur_kernel_size=trial.suggest_categorical("blur_kernel_size", [3, 5, 7]),
        blur_p=trial.suggest_float("blur_p", 0.1, 0.5),
        # Gaussian noise
        noise_std_range=(
            trial.suggest_float("noise_std_low", 1.0, 5.0),
            trial.suggest_float("noise_std_high", 8.0, 20.0),
        ),
        noise_p=trial.suggest_float("noise_p", 0.2, 0.6),
        # Colour jitter
        colour_jitter_p=trial.suggest_float("colour_jitter_p", 0.2, 0.7),
        hue_max_shift=trial.suggest_int("hue_max_shift", 5, 20),
        saturation_range=(
            trial.suggest_float("sat_low", 0.3, 0.8),
            trial.suggest_float("sat_high", 1.2, 1.8),
        ),
    )

    return run_short_training(cfg, epochs=8, trial=trial)


def run_short_training(
    aug_cfg: AugmentationConfig,
    epochs: int,
    trial: optuna.Trial,
) -> float:
    """Run a short training loop and return best val accuracy."""
    train_dataset = LMDBScoreDataset(
        "lmdb/train.lmdb", transform=TrainTransform(aug_cfg)
    )
    val_dataset = LMDBScoreDataset("lmdb/val.lmdb", transform=ValTransform())

    sampler = make_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=64, sampler=sampler, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device(get_device())
    model = build_model(input_size=64).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device=device)

    best_val_acc = 0.0
    for epoch in range(epochs):
        t0 = time.time()
        run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp=True,
            train=True,
        )
        _, val_acc, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp=True,
            train=False,
        )
        best_val_acc = max(best_val_acc, val_acc)
        elapsed = time.time() - t0
        print(f"  epoch={epoch}  val_acc={val_acc:.4f}  time={elapsed:.1f}s")

        # Report to pruner and exit early if this trial is unpromising
        trial.report(best_val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="augmentation_search",
        storage="sqlite:///augmentation_search.db",
        load_if_exists=True,
        pruner=optuna.pruners.PatientPruner(
            optuna.pruners.HyperbandPruner(
                min_resource=5, max_resource=8, reduction_factor=2
            ),
            patience=1,
        ),
    )
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("Best val accuracy:", study.best_value)
    print("Best config:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
