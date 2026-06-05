#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import matplotlib.pyplot as plt

from train_crop_classifier import CACHE_FILES, load_cached_embeddings, load_domain_samples
from train_disease_classifier import BinaryHead, classify_binary_label


TARGET_NAMES = ["healthy", "diseased"]


def load_domain_binary_with_paths(domain: str):
    embeddings = load_cached_embeddings(CACHE_FILES[domain])
    image_paths, class_labels = load_domain_samples(domain)

    if len(class_labels) != embeddings.shape[0]:
        raise ValueError(
            f"[{domain}] label count and embedding count mismatch: "
            f"labels={len(class_labels):,}, embeddings={embeddings.shape[0]:,}"
        )

    labels = [classify_binary_label(domain, label) for label in class_labels]
    keep = np.asarray([label is not None for label in labels], dtype=bool)

    kept_embeddings = embeddings[keep]
    kept_labels = np.asarray([int(label) for label in labels if label is not None], dtype=np.int64)
    kept_paths = [path for path, k in zip(image_paths, keep) if k]
    kept_raw_labels = [label for label, k in zip(class_labels, keep) if k]

    return kept_embeddings, kept_labels, kept_paths, kept_raw_labels


@torch.no_grad()
def predict_cases(model, x, y, paths, raw_labels, domain, batch_size, device, threshold):
    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_labels = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(batch_y.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= threshold).astype(np.int64)

    cases = []
    for path, raw_label, gt, pred, prob in zip(paths, raw_labels, labels, preds, probs):
        correct = int(gt) == int(pred)
        confidence = float(prob if pred == 1 else 1.0 - prob)

        if gt == 1 and pred == 1:
            case_type = "TP_success_disease"
        elif gt == 0 and pred == 0:
            case_type = "TN_success_healthy"
        elif gt == 0 and pred == 1:
            case_type = "FP_failure_healthy_to_disease"
        else:
            case_type = "FN_failure_disease_to_healthy"

        cases.append({
            "domain": domain,
            "image_path": str(path),
            "raw_label": str(raw_label),
            "gt_id": int(gt),
            "gt_name": TARGET_NAMES[int(gt)],
            "pred_id": int(pred),
            "pred_name": TARGET_NAMES[int(pred)],
            "disease_prob": float(prob),
            "confidence": confidence,
            "correct": correct,
            "case_type": case_type,
        })

    return cases


def save_csv(cases, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "image_path",
        "raw_label",
        "gt_id",
        "gt_name",
        "pred_id",
        "pred_name",
        "disease_prob",
        "confidence",
        "correct",
        "case_type",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)


def copy_selected_images(cases, output_dir: Path, top_k: int):
    grouped = {}
    for case in cases:
        grouped.setdefault(case["case_type"], []).append(case)

    for case_type, items in grouped.items():
        items = sorted(items, key=lambda x: x["confidence"], reverse=True)[:top_k]
        case_dir = output_dir / case_type
        case_dir.mkdir(parents=True, exist_ok=True)

        for i, case in enumerate(items, start=1):
            src = Path(case["image_path"])
            suffix = src.suffix.lower()
            dst = case_dir / (
                f"{i:02d}_gt-{case['gt_name']}_pred-{case['pred_name']}"
                f"_conf-{case['confidence']:.3f}{suffix}"
            )
            shutil.copy2(src, dst)


def make_contact_sheet(cases, output_path: Path, title: str, top_k: int):
    selected = sorted(cases, key=lambda x: x["confidence"], reverse=True)[:top_k]
    if not selected:
        return

    cols = min(5, len(selected))
    rows = int(np.ceil(len(selected) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6))
    if rows == 1 and cols == 1:
        axes = np.asarray([axes])
    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, case in zip(axes, selected):
        img = Image.open(case["image_path"]).convert("RGB")
        ax.imshow(img)
        ax.set_title(
            f"{case['case_type']}\n"
            f"GT={case['gt_name']} / Pred={case['pred_name']}\n"
            f"P(disease)={case['disease_prob']:.3f}, conf={case['confidence']:.3f}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/binary_disease_classifier.pt")
    parser.add_argument("--eval-domains", nargs="+", default=["PlantDoc", "PlantWild"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/case_analysis")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    threshold = args.threshold
    if threshold is None:
        threshold = float(ckpt.get("threshold", ckpt_args.get("threshold", 0.5)))

    all_cases = []

    first_x, _, _, _ = load_domain_binary_with_paths(args.eval_domains[0])
    input_dim = first_x.shape[1]

    model = BinaryHead(
        input_dim=input_dim,
        hidden_dim=int(ckpt_args.get("hidden_dim", 0)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])

    for domain in args.eval_domains:
        x, y, paths, raw_labels = load_domain_binary_with_paths(domain)
        cases = predict_cases(
            model=model,
            x=x,
            y=y,
            paths=paths,
            raw_labels=raw_labels,
            domain=domain,
            batch_size=args.batch_size,
            device=device,
            threshold=threshold,
        )
        all_cases.extend(cases)

        domain_dir = Path(args.output_dir) / domain
        save_csv(cases, domain_dir / "all_cases.csv")
        copy_selected_images(cases, domain_dir / "images", args.top_k)

        for case_type in sorted(set(c["case_type"] for c in cases)):
            subset = [c for c in cases if c["case_type"] == case_type]
            make_contact_sheet(
                subset,
                domain_dir / f"{case_type}_top{args.top_k}.png",
                f"{domain} - {case_type}",
                args.top_k,
            )

        n = len(cases)
        fail = sum(not c["correct"] for c in cases)
        success = sum(c["correct"] for c in cases)
        print(f"[{domain}] total={n}, success={success}, failure={fail}")

    output_dir = Path(args.output_dir)
    save_csv(all_cases, output_dir / "all_cases.csv")

    failures = [c for c in all_cases if not c["correct"]]
    successes = [c for c in all_cases if c["correct"]]

    save_csv(sorted(failures, key=lambda x: x["confidence"], reverse=True), output_dir / "failures_sorted.csv")
    save_csv(sorted(successes, key=lambda x: x["confidence"], reverse=True), output_dir / "successes_sorted.csv")

    print(f"saved case analysis to: {output_dir}")


if __name__ == "__main__":
    main()