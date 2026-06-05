#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def draw_heatmap(
    matrix,
    labels,
    title,
    output_path,
):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(
        matrix,
        vmin=np.min(matrix),
        vmax=np.max(matrix),
    )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=30,
        ha="right",
    )

    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{matrix[i,j]:.4f}",
                ha="center",
                va="center",
                fontsize=10,
            )

    cbar = plt.colorbar(im)
    cbar.set_label("Cosine Distance")

    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"saved: {output_path}")


def draw_summary_bar(output_path):
    labels = [
        "PV-PV",
        "PV-PD",
        "PV-PW",
        "PD-PW",
        "Healthy-Disease",
        "Avg Intra-Class",
    ]

    values = [
        0.2854,
        0.5685,
        0.7410,
        0.7514,
        0.5576,
        0.5041,
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(labels, values)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Cosine Distance")
    ax.set_title("Apple Distance Summary")
    ax.set_ylim(0, 0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"saved: {output_path}")


def main():
    output_dir = Path(
        "runs/crop_distance_analysis/apple/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    domain_matrix = np.array([
        [0.2854, 0.5685, 0.7410],
        [0.5681, 0.6161, 0.7514],
        [0.7410, 0.7514, 0.8075],
    ])

    class_matrix = np.array([
        [0.4274, 0.5576],
        [0.5576, 0.5808],
    ])

    draw_heatmap(
        domain_matrix,
        [
            "PlantVillage",
            "PlantDoc",
            "PlantWild",
        ],
        "Apple Domain Distance",
        output_dir / "apple_domain_distance_heatmap.png",
    )

    draw_heatmap(
        class_matrix,
        [
            "Healthy",
            "Diseased",
        ],
        "Apple Class Distance",
        output_dir / "apple_class_distance_heatmap.png",
    )

    draw_summary_bar(
        output_dir / "apple_distance_summary_bar.png"
    )


if __name__ == "__main__":
    main()