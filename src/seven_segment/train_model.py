import argparse
import logging
import os
import pstats
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from scripts.util.build_lmdb import LMDBScoreDataset
from src.model.reader.SevenSegmentScorePreprocessor import (
    PreprocessorConfig,
    SevenSegmentScorePreprocessor,
)

from .data_augmentor import AugmentationConfig, SevenSegmentAugmenter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_CLASSES = 16


@dataclass
class TrainingConfig:
    train_lmdb: str
    val_lmdb: str
    output_dir: str

    input_size: int = 64  # must match PreprocessorConfig.output_size
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # LR scheduler: cosine annealing down to this fraction of initial LR
    lr_min_fraction: float = 0.01

    # Early stopping: halt if val loss does not improve for this many epochs
    early_stopping_patience: int = 8

    # Mixed precision
    use_amp: bool = True

    # Checkpoint: save a checkpoint whenever val loss improves
    save_best_only: bool = True

    # How often to compute and save a confusion matrix (every N epochs)
    confusion_matrix_every_n_epochs: int = 5


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_path}")
    return logger


# ---------------------------------------------------------------------------
# Transform — augmenter + preprocessor composed as a callable
# ---------------------------------------------------------------------------


class TrainTransform:
    """
    Applied in __getitem__ on the DataLoader worker.
    Augments the raw BGR crop, then runs the preprocessor to produce the
    normalised single-channel tensor the model expects.
    """

    def __init__(self, aug_cfg: AugmentationConfig = None):
        self.augmenter = SevenSegmentAugmenter(aug_cfg or AugmentationConfig())
        self.preprocessor = SevenSegmentScorePreprocessor(PreprocessorConfig())
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # HxW uint8 -> 1xHxW float [0,1]
                transforms.Normalize(mean=[0.5], std=[0.5]),  # -> [-1, 1]
            ]
        )

    def __call__(self, image):
        image = self.augmenter.augment(image)
        image = self.preprocessor.process(image)  # -> HxW uint8 grayscale
        return self.to_tensor(image)


class ValTransform:
    """No augmentation — just preprocess and normalise."""

    def __init__(self):
        self.preprocessor = SevenSegmentScorePreprocessor(PreprocessorConfig())
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __call__(self, image):
        image = self.preprocessor.process(image)
        return self.to_tensor(image)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(input_size: int) -> nn.Module:
    """
    MobileNetV2 pretrained on ImageNet, adapted for:
      - Single-channel (grayscale) input via a learned 1->3 channel projection
      - 16-class output head
    The first conv layer is replaced rather than averaging channels so the
    model can learn an optimal grayscale-to-feature mapping rather than
    assuming equal channel weighting.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Replace first conv: 3-channel -> 1-channel input
    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    # Initialise by averaging the pretrained RGB weights across channels
    with torch.no_grad():
        model.features[0][0].weight = nn.Parameter(
            old_conv.weight.mean(dim=1, keepdim=True)
        )

    # Replace classifier head for 16 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model


# ---------------------------------------------------------------------------
# Weighted sampler for class imbalance
# ---------------------------------------------------------------------------


def make_weighted_sampler(dataset: LMDBScoreDataset) -> WeightedRandomSampler:
    """
    Oversample minority classes so each class appears roughly equally
    per epoch, without discarding any majority class samples.
    """
    class_counts = dataset.class_counts
    total = sum(class_counts.values())
    class_weights = {
        cls: total / count for cls, count in class_counts.items() if count > 0
    }

    labels = dataset.read_all_labels()
    sample_weights = [class_weights.get(label, 1.0) for label in labels]

    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


# ---------------------------------------------------------------------------
# Training and validation steps
# ---------------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    train: bool,
    collect_preds: bool = False,
) -> tuple[float, float, np.ndarray | None, np.ndarray | None]:
    """
    Returns (loss, accuracy, all_preds, all_labels).
    all_preds and all_labels are only populated when collect_preds=True,
    otherwise None — collecting them every epoch adds minor overhead so
    we only do it when a confusion matrix is needed.
    """
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    all_preds = [] if collect_preds else None
    all_labels = [] if collect_preds else None

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=use_amp, device_type=str(device)):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            if collect_preds:
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    preds_out = np.concatenate(all_preds) if collect_preds else None
    labels_out = np.concatenate(all_labels) if collect_preds else None

    return total_loss / total, correct / total, preds_out, labels_out


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    present_classes: list[int],
) -> np.ndarray:
    """
    Compute a confusion matrix restricted to classes present in the val set.
    Rows = true class, columns = predicted class.
    """
    n = len(present_classes)
    idx_map = {cls: i for i, cls in enumerate(present_classes)}
    matrix = np.zeros((n, n), dtype=np.int64)
    for true, pred in zip(labels, preds):
        if true in idx_map:
            matrix[idx_map[true], idx_map.get(pred, -1)] += 1 if pred in idx_map else 0
    return matrix


def log_confusion_matrix(
    matrix: np.ndarray,
    present_classes: list[int],
    epoch: int,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Log the confusion matrix as a text table and save a heatmap image."""
    n = len(present_classes)
    labels = [str(c) for c in present_classes]

    # Text log — per-class accuracy on the diagonal
    logger.info(f"  Confusion matrix (epoch {epoch}, rows=true, cols=pred):")
    header = "       " + "  ".join(f"{l:>3}" for l in labels)
    logger.info(header)
    for i, row in enumerate(matrix):
        row_total = row.sum()
        row_acc = matrix[i, i] / row_total if row_total > 0 else 0.0
        row_str = "  ".join(f"{v:>3}" for v in row)
        logger.info(f"  [{labels[i]:>3}]  {row_str}   acc={row_acc:.2f}")

    # Per-class accuracy summary
    logger.info("  Per-class accuracy:")
    for i, cls in enumerate(present_classes):
        row_total = matrix[i].sum()
        acc = matrix[i, i] / row_total if row_total > 0 else 0.0
        logger.info(f"    class {cls:2d}: {acc:.3f}  ({matrix[i,i]}/{row_total})")

    # Heatmap — normalise each row so colour represents recall per class
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.where(row_sums > 0, matrix / row_sums, 0).astype(float)

    fig, ax = plt.subplots(figsize=(len(present_classes), len(present_classes) - 1))
    im = ax.imshow(norm_matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — epoch {epoch} (normalised by row)")
    plt.colorbar(im, ax=ax)

    # Annotate cells with raw counts
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm_matrix[i, j] > 0.6 else "black",
            )

    path = os.path.join(output_dir, f"confusion_matrix_epoch_{epoch:03d}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
    logger.info(f"  Confusion matrix saved to {path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg: TrainingConfig) -> None:
    logger = setup_logging(cfg.output_dir)
    logger.info(f"Config: {cfg}")

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Datasets
    train_dataset = LMDBScoreDataset(cfg.train_lmdb, transform=TrainTransform())
    val_dataset = LMDBScoreDataset(cfg.val_lmdb, transform=ValTransform())
    logger.info(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")
    logger.info(f"Train class counts: {train_dataset.class_counts}")
    logger.info(f"Val class counts:   {val_dataset.class_counts}")

    # Classes present in the val set — confusion matrix is restricted to these
    val_present_classes = sorted(
        cls for cls, count in val_dataset.class_counts.items() if count > 0
    )
    logger.info(f"Val present classes: {val_present_classes}")

    sampler = make_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Model
    model = build_model(cfg.input_size).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.lr * cfg.lr_min_fraction,
    )
    scaler = GradScaler(enabled=cfg.use_amp, device=device)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = os.path.join(cfg.output_dir, "best_model.pt")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        do_confusion = (epoch % cfg.confusion_matrix_every_n_epochs == 0) or epoch == 1

        train_loss, train_acc, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            cfg.use_amp,
            train=True,
            collect_preds=False,
        )
        val_loss, val_acc, val_preds, val_labels = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            scaler,
            device,
            cfg.use_amp,
            train=False,
            collect_preds=do_confusion,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={lr_now:.2e}  time={elapsed:.1f}s"
        )

        # Confusion matrix
        if do_confusion:
            matrix = compute_confusion_matrix(
                val_preds, val_labels, val_present_classes
            )
            log_confusion_matrix(
                matrix, val_present_classes, epoch, cfg.output_dir, logger
            )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )
            logger.info(f"  -> Saved checkpoint (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  -> No improvement ({epochs_without_improvement}/{cfg.early_stopping_patience})"
            )

        # Early stopping
        if epochs_without_improvement >= cfg.early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    logger.info(f"Best checkpoint saved to {checkpoint_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train seven-segment score classifier")
    parser.add_argument("train_lmdb", help="Path to training LMDB")
    parser.add_argument("val_lmdb", help="Path to validation LMDB")
    parser.add_argument("output_dir", help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument(
        "--cm-every",
        type=int,
        default=5,
        help="Save confusion matrix every N epochs (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":

    def main():
        args = parse_args()
        cfg = TrainingConfig(
            train_lmdb=args.train_lmdb,
            val_lmdb=args.val_lmdb,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            use_amp=not args.no_amp,
            confusion_matrix_every_n_epochs=args.cm_every,
        )
        train(cfg)

    import cProfile

    cProfile.run("main()", "profile.stats")
    stats = pstats.Stats("profile.stats")
    stats.strip_dirs()
    stats.sort_stats("tottime")
    stats.print_stats(20)
