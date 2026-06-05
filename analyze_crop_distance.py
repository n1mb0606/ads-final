#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from train_crop_classifier import (
    CACHE_FILES,
    load_cached_embeddings,
    load_domain_samples,
    extract_crop_name,
)

from train_disease_classifier import classify_binary_label


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


def load_crop_binary_domain(domain: str, target_crop: str):
    embeddings = load_cached_embeddings(CACHE_FILES[domain])
    image_paths, class_labels = load_domain_samples(domain)

    if len(class_labels) != embeddings.shape[0]:
        raise ValueError(
            f"[{domain}] label count and embedding count mismatch: "
            f"labels={len(class_labels):,}, embeddings={embeddings.shape[0]:,}"
        )

    target_crop = target_crop.lower().strip()

    keep_indices = []
    labels = []
    crops = []
    raw_labels = []

    for idx, raw_label in enumerate(class_labels):
        crop = extract_crop_name(domain, raw_label)
        binary_label = classify_binary_label(domain, raw_label)

        if crop != target_crop:
            continue

        if binary_label is None:
            continue

        keep_indices.append(idx)
        labels.append(int(binary_label))
        crops.append(crop)
        raw_labels.append(str(raw_label))

    if keep_indices:
        x = embeddings[np.asarray(keep_indices, dtype=np.int64)]
        y = np.asarray(labels, dtype=np.int64)
    else:
        x = np.empty((0, embeddings.shape[1]), dtype=np.float32)
        y = np.asarray([], dtype=np.int64)

    stats = {
        "domain": domain,
        "crop": target_crop,
        "used": int(x.shape[0]),
        "healthy": int((y == 0).sum()),
        "diseased": int((y == 1).sum()),
    }

    return x, y, stats


def save_matrix_csv(matrix, labels, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)

        for label, row in zip(labels, matrix):
            writer.writerow(
                [label]
                + [
                    f"{v:.6f}" if not np.isnan(v) else "nan"
                    for v in row
                ]
            )


def save_rows_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(matrix, labels, title, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(matrix)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            text = "nan" if np.isnan(value) else f"{value:.4f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    ax.set_title(title, fontsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Distance", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def build_domain_distance(domain_to_x, domains, max_samples, seed):
    """
    특정 crop만 남긴 상태에서
    healthy + diseased를 합쳐 domain끼리 비교.
    """
    n = len(domains)
    matrix = np.zeros((n, n), dtype=np.float64)
    rows = []

    for i, d1 in enumerate(domains):
        x1 = domain_to_x[d1]

        for j, d2 in enumerate(domains):
            x2 = domain_to_x[d2]

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
                "group_type": "crop_domain",
                "domain_a": d1,
                "domain_b": d2,
                "relation": relation,
                "cosine_distance": dist,
                "n_a": int(x1.shape[0]),
                "n_b": int(x2.shape[0]),
            })

    return matrix, rows


def build_class_distance(all_x, all_y, max_samples, seed):
    """
    특정 crop만 남긴 상태에서
    domain을 합쳐 healthy vs diseased 비교.
    """
    class_ids = [0, 1]
    class_labels = [CLASS_NAMES[c] for c in class_ids]

    matrix = np.zeros((2, 2), dtype=np.float64)
    rows = []

    for i, c1 in enumerate(class_ids):
        x1 = all_x[all_y == c1]

        for j, c2 in enumerate(class_ids):
            x2 = all_x[all_y == c2]

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
                "group_type": "crop_class",
                "class_a": CLASS_NAMES[c1],
                "class_b": CLASS_NAMES[c2],
                "relation": relation,
                "cosine_distance": dist,
                "n_a": int(x1.shape[0]),
                "n_b": int(x2.shape[0]),
            })

    return matrix, class_labels, rows


def build_domain_class_distance(domain_to_x, domain_to_y, domains, max_samples, seed):
    """
    특정 crop만 남긴 상태에서
    domain + class 조합끼리 비교.

    예:
    PlantVillage_healthy vs PlantWild_healthy
    PlantVillage_diseased vs PlantWild_diseased
    """
    group_names = []
    group_features = []

    for domain in domains:
        for class_id in [0, 1]:
            group_names.append(f"{domain}_{CLASS_NAMES[class_id]}")
            group_features.append(domain_to_x[domain][domain_to_y[domain] == class_id])

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
                "group_type": "crop_domain_class",
                "group_a": name1,
                "group_b": name2,
                "relation": relation,
                "cosine_distance": dist,
                "n_a": int(x1.shape[0]),
                "n_b": int(x2.shape[0]),
            })

    return matrix, group_names, rows


def print_available_crops(domains):
    print("\n===== Available crops by domain =====")

    for domain in domains:
        _, class_labels = load_domain_samples(domain)
        crops = sorted(set(extract_crop_name(domain, label) for label in class_labels))
        print(f"\n[{domain}]")
        print(", ".join(crops))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--crop",
        required=False,
        default=None,
        help="분석할 작물명. 예: tomato, potato, grape, corn, apple",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["PlantVillage", "PlantDoc", "PlantWild"],
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="각 group별 최대 샘플 수. 0이면 전체 사용.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="runs/crop_distance_analysis",
    )
    parser.add_argument(
        "--list-crops",
        action="store_true",
        help="도메인별 추출 가능한 crop 이름만 출력하고 종료.",
    )

    args = parser.parse_args()

    if args.list_crops:
        print_available_crops(args.domains)
        return

    if args.crop is None:
        raise ValueError("--crop을 지정해야 함. 예: --crop tomato")

    crop = args.crop.lower().strip()
    output_dir = Path(args.output_dir) / crop.replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_to_x = {}
    domain_to_y = {}
    stats = []

    for domain in args.domains:
        x, y, s = load_crop_binary_domain(domain, crop)
        domain_to_x[domain] = x
        domain_to_y[domain] = y
        stats.append(s)

        print(
            f"{domain}: crop={crop}, used={s['used']:,}, "
            f"healthy={s['healthy']:,}, diseased={s['diseased']:,}"
        )

    non_empty_domains = [d for d in args.domains if domain_to_x[d].shape[0] > 0]

    if len(non_empty_domains) < 2:
        raise ValueError(
            f"'{crop}' 샘플이 2개 이상 도메인에 존재해야 domain 비교 가능. "
            f"현재 non-empty domains={non_empty_domains}"
        )

    all_x = np.concatenate([domain_to_x[d] for d in non_empty_domains], axis=0)
    all_y = np.concatenate([domain_to_y[d] for d in non_empty_domains], axis=0)

    domain_matrix, domain_rows = build_domain_distance(
        domain_to_x,
        non_empty_domains,
        args.max_samples,
        args.seed,
    )

    class_matrix, class_labels, class_rows = build_class_distance(
        all_x,
        all_y,
        args.max_samples,
        args.seed,
    )

    domain_class_matrix, domain_class_labels, domain_class_rows = build_domain_class_distance(
        domain_to_x,
        domain_to_y,
        non_empty_domains,
        args.max_samples,
        args.seed,
    )

    save_rows_csv(stats, output_dir / "dataset_stats.csv")

    save_matrix_csv(
        domain_matrix,
        non_empty_domains,
        output_dir / "crop_domain_distance_matrix.csv",
    )
    save_rows_csv(
        domain_rows,
        output_dir / "crop_domain_distance_pairs.csv",
    )
    save_heatmap(
        domain_matrix,
        non_empty_domains,
        f"Domain Distance for Crop: {crop}\nHealthy and Diseased Merged",
        output_dir / "crop_domain_distance_heatmap.png",
    )

    save_matrix_csv(
        class_matrix,
        class_labels,
        output_dir / "crop_class_distance_matrix.csv",
    )
    save_rows_csv(
        class_rows,
        output_dir / "crop_class_distance_pairs.csv",
    )
    save_heatmap(
        class_matrix,
        class_labels,
        f"Class Distance for Crop: {crop}\nAll Domains Merged",
        output_dir / "crop_class_distance_heatmap.png",
    )

    save_matrix_csv(
        domain_class_matrix,
        domain_class_labels,
        output_dir / "crop_domain_class_distance_matrix.csv",
    )
    save_rows_csv(
        domain_class_rows,
        output_dir / "crop_domain_class_distance_pairs.csv",
    )
    save_heatmap(
        domain_class_matrix,
        domain_class_labels,
        f"Domain-Class Distance for Crop: {crop}",
        output_dir / "crop_domain_class_distance_heatmap.png",
    )

    print("\n===== Summary: Domain Distance =====")
    for i, d1 in enumerate(non_empty_domains):
        for j, d2 in enumerate(non_empty_domains):
            print(f"{d1:12s} ↔ {d2:12s}: {domain_matrix[i, j]:.4f}")

    print("\n===== Summary: Class Distance =====")
    for i, c1 in enumerate(class_labels):
        for j, c2 in enumerate(class_labels):
            print(f"{c1:8s} ↔ {c2:8s}: {class_matrix[i, j]:.4f}")

    # Class Separation Summary를 위해 각 클래스의 샘플에서 centroid까지의 평균 거리를 계산합니다.
    intra_dists = []
    for i, c_name in enumerate(class_labels):
        c_id = 0 if c_name == "healthy" else 1
        x_class = all_x[all_y == c_id]
        if len(x_class) > 0:
            d = average_distance_to_centroid(x_class, max_samples=args.max_samples, seed=args.seed + i)
            intra_dists.append(d)

    intra_class = np.mean(intra_dists) if intra_dists else float("nan")
    inter_class = class_matrix[0, 1]
    ratio = inter_class / intra_class if intra_class > 0 else float("nan")

    print("\n===== Class Separation Summary =====")
    print(f"Average intra-class compactness (mean dist to centroid) : {intra_class:.4f}")
    print(f"Inter-class centroid distance                          : {inter_class:.4f}")
    print(f"Separation ratio (inter / intra)                        : {ratio:.4f}")

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()