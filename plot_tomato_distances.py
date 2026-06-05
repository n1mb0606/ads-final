#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_heatmap(matrix, labels, title, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.4f}",
                ha="center",
                va="center",
                fontsize=10,
            )

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine Distance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"saved: {output_path}")


def save_summary_bar(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [
        "PV-PV\nwithin",
        "PV-PD",
        "PV-PW",
        "PD-PW",
        "Healthy-Disease",
        "Avg Intra-Class",
    ]

    values = [
        0.3987,
        0.6186,
        0.6714,
        0.6242,
        0.4766,
        0.4250,
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values)

    ax.set_ylabel("Cosine Distance")
    ax.set_title("Tomato Distance Summary")
    ax.set_ylim(0, max(values) * 1.2)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"saved: {output_path}")


def main():
    output_dir = Path("runs/crop_distance_analysis/tomato/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_labels = [
        "PlantVillage",
        "PlantDoc",
        "PlantWild",
    ]

    domain_matrix = np.array([
        [0.3987, 0.6186, 0.6714],
        [0.6174, 0.5754, 0.6242],
        [0.6676, 0.6242, 0.6539],
    ])

    class_labels = [
        "Healthy",
        "Diseased",
    ]

    class_matrix = np.array([
        [0.3894, 0.4766],
        [0.4766, 0.4607],
    ])

    save_heatmap(
        domain_matrix,
        domain_labels,
        "Tomato Domain Distance\nHealthy and Diseased Samples Merged",
        output_dir / "tomato_domain_distance_heatmap.png",
    )

    save_heatmap(
        class_matrix,
        class_labels,
        "Tomato Class Distance\nAll Domains Merged",
        output_dir / "tomato_class_distance_heatmap.png",
    )

    save_summary_bar(
        output_dir / "tomato_distance_summary_bar.png",
    )


if __name__ == "__main__":
    main()