import argparse
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.momentum_graph.util.file_names import LAB_VALUES_CSV


def get_parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise LAB value distributions of score lights"
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="Path to output folders for intermediate/final products",
    )
    return parser.parse_args()


def main():
    args = get_parse_args()
    folders = args.folders

    input_csv_paths = [path.join(folder, LAB_VALUES_CSV) for folder in folders]

    fig, ax = plt.subplots(figsize=(6, 6))

    label_cfg = {
        "red": dict(cmap="Reds", marker="x"),
        "green": dict(cmap="Greens", marker="o"),
    }

    for csv_path in input_csv_paths:
        df = pd.read_csv(csv_path)
        folder_name = path.basename(path.dirname(csv_path))

        for label, cfg in label_cfg.items():
            sub = df[df["label"] == label]
            if sub.empty:
                continue

            # density
            ax.hexbin(
                sub["a"],
                sub["b"],
                gridsize=60,
                cmap=cfg["cmap"],
                mincnt=1,
                alpha=0.4,
            )

            # mean per file + label
            mu_a, mu_b = sub[["a", "b"]].mean()
            ax.scatter(
                mu_a,
                mu_b,
                c="black",
                s=60,
                marker=cfg["marker"],
            )
            ax.annotate(
                f"{folder_name} ({label})",
                (mu_a, mu_b),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title("a–b Distributions (LAB)")
    plt.show()


if __name__ == "__main__":
    main()
