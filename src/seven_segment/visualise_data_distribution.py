import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def count_classes(dataset_dirs: list[str]) -> dict[str, dict[int, int]]:
    """Count images per class for each dataset directory."""
    counts = {}
    for d in dataset_dirs:
        label = os.path.basename(os.path.abspath(d))
        counts[label] = {}
        for class_id in range(16):
            class_dir = os.path.join(d, str(class_id))
            if os.path.isdir(class_dir):
                counts[label][class_id] = len(
                    [f for f in os.listdir(class_dir) if f.lower().endswith(".png")]
                )
            else:
                counts[label][class_id] = 0
    return counts


def combined_counts(counts: dict[str, dict[int, int]]) -> dict[int, int]:
    combined = defaultdict(int)
    for class_counts in counts.values():
        for c, n in class_counts.items():
            combined[c] += n
    return dict(combined)


def plot_single(ax, class_counts: dict[int, int], title: str) -> None:
    classes = list(range(16))
    values = [class_counts[c] for c in classes]
    total = sum(values)
    bars = ax.bar([str(c) for c in classes], values)
    ax.set_title(f"{title}\n({total} total crops)")
    ax.set_xlabel("Score class")
    ax.set_ylabel("Image count")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            str(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_distribution(counts: dict[str, dict[int, int]]) -> None:
    n_dirs = len(counts)
    # Individual plots + one combined column
    n_cols = n_dirs + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)

    for ax, (label, class_counts) in zip(axes, counts.items()):
        plot_single(ax, class_counts, label)

    # Combined column shares the y-axis scale so imbalance is visually comparable
    plot_single(axes[-1], combined_counts(counts), "COMBINED")

    plt.suptitle("Class distribution per dataset folder", fontsize=13)
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise class distribution of collected datasets"
    )
    parser.add_argument(
        "dataset_dirs", nargs="+", help="One or more dataset root folders"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    counts = count_classes(args.dataset_dirs)

    for label, class_counts in counts.items():
        total = sum(class_counts.values())
        print(f"\n{label} ({total} total):")
        for c, n in class_counts.items():
            print(
                f"  class {c:2d}: {n:6d}  ({100*n/total:.1f}%)"
                if total
                else f"  class {c:2d}: 0"
            )

    combined = combined_counts(counts)
    total = sum(combined.values())
    print(f"\nCOMBINED ({total} total):")
    for c, n in combined.items():
        print(
            f"  class {c:2d}: {n:6d}  ({100*n/total:.1f}%)"
            if total
            else f"  class {c:2d}: 0"
        )

    plot_distribution(counts)
