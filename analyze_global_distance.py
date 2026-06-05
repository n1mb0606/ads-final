#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


from train_disease_classifier import load_domain_binary_arrays


CLASS_NAMES = {
    0: "healthy",
    1: "diseased",
}


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def sample_rows(x: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x

    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_samples, replace=False)
    return x[idx]


def cosine_distance_between_sets(
    a: np.ndarray,
    b: np.ndarray,
    max_samples: int,
    seed: int,
) -> float:
    """
    두 feature set의 centroid 간 cosine distance 계산.
    distance = 1 - cosine_similarity(centroid_a, centroid_b)

    a, b가 너무 크면 max_samples만큼 sampling해서 계산.
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")

    a = sample_rows(a, max_samples, seed)
    b = sample_rows(b, max_samples, seed + 1)

    a = l2_normalize(a)
    b = l2_normalize(b)

    centroid_a = a.mean(axis=0, keepdims=True)
    centroid_b = b.mean(axis=0, keepdims=True)

    centroid_a = l2_normalize(centroid_a)
    centroid_b = l2_normalize(centroid_b)

    sim = float(np.sum(centroid_a * centroid_b))
    dist = 1.0 - sim

    return dist


def cosine_distance_within_set(
    x: np.ndarray,
    max_samples: int,
    seed: int,
) -> float:
    """
    Centroid 간 거리 분석에서 자기 자신(동일 도메인/클래스)과의 거리는 0.0입니다.
    """
    return 0.0


def average_distance_to_centroid(
    x: np.ndarray,
    max_samples: int,
    seed: int,
) -> float:
    """
    샘플들과 해당 군집의 centroid 간의 평균 cosine distance 계산 (군집 내 분산/밀집도 지표).
    """
    if x.shape[0] == 0:
        return float("nan")

    x = sample_rows(x, max_samples, seed)
    x = l2_normalize(x)

    centroid = x.mean(axis=0, keepdims=True)
    centroid = l2_normalize(centroid)

    sim = x @ centroid.T
    dist = 1.0 - sim

    return float(dist.mean())


def load_all_domains(domains: list[str]):
    all_x = []
    all_y = []
    all_d = []

    stats = []

    for domain in domains:
        x, y, skipped, counts = load_domain_binary_arrays(domain)

        all_x.append(x)
        all_y.append(y)
        all_d.extend([domain] * x.shape[0])

        stats.append({
            "domain": domain,
            "used": int(x.shape[0]),
            "healthy": int(counts["healthy"]),
            "diseased": int(counts["diseased"]),
            "unknown": int(counts["unknown"]),
        })

        print(
            f"{domain}: used={x.shape[0]:,}, "
            f"healthy={counts['healthy']:,}, "
            f"diseased={counts['diseased']:,}, "
            f"unknown={counts['unknown']:,}"
        )

    return (
        np.concatenate(all_x, axis=0),
        np.concatenate(all_y, axis=0),
        np.asarray(all_d, dtype=object),
        stats,
    )


def save_matrix_csv(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)

        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{v:.6f}" if not np.isnan(v) else "nan" for v in row])


def save_pair_csv(
    rows: list[dict],
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            text = "nan" if np.isnan(value) else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def analyze_domain_distance(
    x: np.ndarray,
    domains_arr: np.ndarray,
    domain_names: list[str],
    max_samples: int,
    seed: int,
):
    """
    healthy/disease를 합쳐서 domain끼리 비교.

    예:
    PlantVillage 전체 vs PlantDoc 전체
    PlantVillage 전체 vs PlantWild 전체
    """
    n = len(domain_names)
    matrix = np.zeros((n, n), dtype=np.float64)
    rows = []

    for i, d1 in enumerate(domain_names):
        x1 = x[domains_arr == d1]

        for j, d2 in enumerate(domain_names):
            x2 = x[domains_arr == d2]

            if i == j:
                dist = cosine_distance_within_set(
                    x1,
                    max_samples=max_samples,
                    seed=seed + i * 100 + j,
                )
                relation = "within_domain"
            else:
                dist = cosine_distance_between_sets(
                    x1,
                    x2,
                    max_samples=max_samples,
                    seed=seed + i * 100 + j,
                )
                relation = "between_domain"

            matrix[i, j] = dist

            rows.append({
                "group_type": "domain",
                "group_a": d1,
                "group_b": d2,
                "relation": relation,
                "cosine_distance": dist,
                "n_a": int(x1.shape[0]),
                "n_b": int(x2.shape[0]),
            })

    return matrix, rows


def analyze_class_distance(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
):
    """
    domain을 합쳐서 healthy vs diseased 비교.

    예:
    전체 Healthy vs 전체 Diseased
    """
    class_ids = [0, 1]
    labels = [CLASS_NAMES[c] for c in class_ids]

    n = len(class_ids)
    matrix = np.zeros((n, n), dtype=np.float64)
    rows = []

    for i, c1 in enumerate(class_ids):
        x1 = x[y == c1]

        for j, c2 in enumerate(class_ids):
            x2 = x[y == c2]

            if i == j:
                dist = cosine_distance_within_set(
                    x1,
                    max_samples=max_samples,
                    seed=seed + i * 100 + j,
                )
                relation = "within_class"
            else:
                dist = cosine_distance_between_sets(
                    x1,
                    x2,
                    max_samples=max_samples,
                    seed=seed + i * 100 + j,
                )
                relation = "between_class"

            matrix[i, j] = dist

            rows.append({
                "group_type": "class",
                "group_a": CLASS_NAMES[c1],
                "group_b": CLASS_NAMES[c2],
                "relation": relation,
                "cosine_distance": dist,
                "n_a": int(x1.shape[0]),
                "n_b": int(x2.shape[0]),
            })

    return matrix, labels, rows


def analyze_domain_class_distance(
    x: np.ndarray,
    y: np.ndarray,
    domains_arr: np.ndarray,
    domain_names: list[str],
    max_samples: int,
    seed: int,
):
    """
    추가 분석:
    domain + class 조합끼리 비교.

    예:
    PlantVillage healthy vs PlantWild healthy
    PlantVillage diseased vs PlantWild diseased
    PlantVillage healthy vs PlantVillage diseased
    """
    group_names = []
    group_features = []

    for domain in domain_names:
        for class_id in [0, 1]:
            mask = (domains_arr == domain) & (y == class_id)
            group_names.append(f"{domain}_{CLASS_NAMES[class_id]}")
            group_features.append(x[mask])

    n = len(group_names)
    matrix = np.zeros((n, n), dtype=np.float64)
    rows = []

    for i, name1 in enumerate(group_names):
        x1 = group_features[i]

        for j, name2 in enumerate(group_names):
            x2 = group_features[j]

            if i == j:
                dist = cosine_distance_within_set(
                    x1,
                    max_samples=max_samples,
                    seed=seed + i * 100 + j,
                )
                relation = "within_group"
            else:
                dist = cosine_distance_between_sets(
                    x1,
                    x2,
                    max_samples=max_samples,
                    seed=seed + i * 100 + j,
                )
                relation = "between_group"

            matrix[i, j] = dist

            rows.append({
                "group_type": "domain_class",
                "group_a": name1,
                "group_b": name2,
                "relation": relation,
                "cosine_distance": dist,
                "n_a": int(x1.shape[0]),
                "n_b": int(x2.shape[0]),
            })

    return matrix, group_names, rows


def print_summary(
    domain_matrix: np.ndarray,
    domain_names: list[str],
    class_matrix: np.ndarray,
    class_names: list[str],
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
):
    print("\n===== Domain Distance (Centroid): healthy + diseased merged =====")
    for i, d1 in enumerate(domain_names):
        for j, d2 in enumerate(domain_names):
            print(f"{d1:12s} ↔ {d2:12s}: {domain_matrix[i, j]:.4f}")

    print("\n===== Class Distance (Centroid): domains merged =====")
    for i, c1 in enumerate(class_names):
        for j, c2 in enumerate(class_names):
            print(f"{c1:8s} ↔ {c2:8s}: {class_matrix[i, j]:.4f}")

    # Class Separation Summary를 위해 각 클래스의 샘플에서 centroid까지의 평균 거리를 계산합니다.
    intra_dists = []
    for i, c_name in enumerate(class_names):
        c_id = 0 if c_name == "healthy" else 1
        x_class = x[y == c_id]
        if len(x_class) > 0:
            d = average_distance_to_centroid(x_class, max_samples=max_samples, seed=seed + i)
            intra_dists.append(d)

    intra_class = np.mean(intra_dists) if intra_dists else float("nan")
    inter_class = class_matrix[0, 1]
    ratio = inter_class / intra_class if intra_class > 0 else float("nan")

    print("\n===== Class Separation Summary =====")
    print(f"Average intra-class compactness (mean dist to centroid) : {intra_class:.4f}")
    print(f"Inter-class centroid distance                          : {inter_class:.4f}")
    print(f"Separation ratio (inter / intra)                        : {ratio:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--domains",
        nargs="+",
        default=["PlantVillage", "PlantDoc", "PlantWild"],
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="각 group마다 최대 샘플 수. 너무 크면 메모리 많이 씀. 0이면 전체 사용.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        default="runs/distance_analysis",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    x, y, domains_arr, stats = load_all_domains(args.domains)

    print(f"\nTotal used samples: {x.shape[0]:,}")
    print(f"Embedding dim: {x.shape[1]:,}")

    domain_matrix, domain_rows = analyze_domain_distance(
        x=x,
        domains_arr=domains_arr,
        domain_names=args.domains,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    class_matrix, class_names, class_rows = analyze_class_distance(
        x=x,
        y=y,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    domain_class_matrix, domain_class_names, domain_class_rows = analyze_domain_class_distance(
        x=x,
        y=y,
        domains_arr=domains_arr,
        domain_names=args.domains,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    print_summary(
        domain_matrix=domain_matrix,
        domain_names=args.domains,
        class_matrix=class_matrix,
        class_names=class_names,
        x=x,
        y=y,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    save_matrix_csv(
        domain_matrix,
        args.domains,
        output_dir / "domain_distance_matrix.csv",
    )
    save_pair_csv(
        domain_rows,
        output_dir / "domain_distance_pairs.csv",
    )
    save_heatmap(
        domain_matrix,
        args.domains,
        "Domain Distance: Healthy + Diseased Merged",
        output_dir / "domain_distance_heatmap.png",
    )

    save_matrix_csv(
        class_matrix,
        class_names,
        output_dir / "class_distance_matrix.csv",
    )
    save_pair_csv(
        class_rows,
        output_dir / "class_distance_pairs.csv",
    )
    save_heatmap(
        class_matrix,
        class_names,
        "Class Distance: Domains Merged",
        output_dir / "class_distance_heatmap.png",
    )

    save_matrix_csv(
        domain_class_matrix,
        domain_class_names,
        output_dir / "domain_class_distance_matrix.csv",
    )
    save_pair_csv(
        domain_class_rows,
        output_dir / "domain_class_distance_pairs.csv",
    )
    save_heatmap(
        domain_class_matrix,
        domain_class_names,
        "Domain-Class Distance",
        output_dir / "domain_class_distance_heatmap.png",
    )

    save_pair_csv(
        stats,
        output_dir / "dataset_stats.csv",
    )

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()