#!/usr/bin/env python3
"""Train a binary healthy-vs-diseased classifier on cached DINOv3 embeddings.

Recommended first experiment:
    python train_disease_classifier.py \
        --train-domains PlantVillage \
        --eval-domains PlantDoc PlantWild

Target convention:
    0 = healthy / normal leaf
    1 = diseased
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_crop_classifier import (
    CACHE_FILES,
    format_duration,
    load_cached_embeddings,
    load_domain_samples,
    split_train_val,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DISEASE_KEYWORDS = {
    'alternaria',
    'anthracnose',
    'bacterial',
    'black rot',
    'blight',
    'blossom end rot',
    'canker',
    'cavity spot',
    'cercospora',
    'curl',
    'downy mildew',
    'early blight',
    'esca',
    'gray leaf spot',
    'gray mold',
    'greening',
    'halo blight',
    'late blight',
    'leaf blight',
    'leaf curl',
    'leaf mold',
    'leaf scorch',
    'leaf spot',
    'leafroll',
    'mildew',
    'mold',
    'mosaic',
    'northern leaf blight',
    'panama disease',
    'powdery mildew',
    'pocket disease',
    'rot',
    'rust',
    'scab',
    'scorch',
    'septoria',
    'sheath blight',
    'smut',
    'spider mites',
    'spot',
    'target spot',
    'virus',
    'wilt',
    'yellow leaf curl',
}

HEALTHY_KEYWORDS = {
    'healthy',
    'normal',
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_label(label: str) -> str:
    text = str(label).lower().strip()
    for ch in ['_', '-', ',', '(', ')']:
        text = text.replace(ch, ' ')
    return ' '.join(text.split())


def contains_keyword(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def classify_single_label(domain: str, class_label: str) -> int | None:
    text = normalize_label(class_label)

    if domain == 'PlantVillage':
        parts = text.split('___')
        disease_part = parts[-1] if len(parts) > 1 else text
        return 0 if 'healthy' in disease_part else 1

    if contains_keyword(text, HEALTHY_KEYWORDS):
        return 0
    if contains_keyword(text, DISEASE_KEYWORDS):
        return 1

    # PlantDoc and PlantWild use labels such as "Cherry leaf" or "apple leaf"
    # for healthy leaves. Disease labels usually include a disease keyword.
    if text.endswith(' leaf') or text == 'leaf':
        return 0

    return None


def classify_binary_label(domain: str, class_label: str) -> int | None:
    parts = [part.strip() for part in str(class_label).split('|') if part.strip()]
    if not parts:
        return None

    decisions = [classify_single_label(domain, part) for part in parts]
    if any(decision == 1 for decision in decisions):
        return 1
    if all(decision == 0 for decision in decisions):
        return 0
    return None


class BinaryEmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.from_numpy(embeddings.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.labels[idx]


def load_domain_binary_arrays(domain: str) -> tuple[np.ndarray, np.ndarray, int, dict[str, int]]:
    embeddings = load_cached_embeddings(CACHE_FILES[domain])
    _, class_labels = load_domain_samples(domain)
    if len(class_labels) != embeddings.shape[0]:
        raise ValueError(
            f'[{domain}] label count and embedding count mismatch: '
            f'labels={len(class_labels):,}, embeddings={embeddings.shape[0]:,}'
        )

    labels = [classify_binary_label(domain, label) for label in class_labels]
    keep = np.asarray([label is not None for label in labels], dtype=bool)
    y = np.asarray([int(label) for label in labels if label is not None], dtype=np.int64)
    skipped = int((~keep).sum())
    counts = {
        'healthy': int((y == 0).sum()),
        'diseased': int((y == 1).sum()),
        'unknown': skipped,
    }
    return embeddings[keep], y, skipped, counts


def concat_domains(domains: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    xs = []
    ys = []
    domain_labels = []
    skipped_total = 0

    for domain in domains:
        x, y, skipped, counts = load_domain_binary_arrays(domain)
        xs.append(x)
        ys.append(y)
        domain_labels.extend([domain] * x.shape[0])
        skipped_total += skipped
        print(
            f'{domain}: used={x.shape[0]:,}, healthy={counts["healthy"]:,}, '
            f'diseased={counts["diseased"]:,}, skipped_unknown={skipped:,}',
            flush=True,
        )

    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.asarray(domain_labels, dtype=object),
        skipped_total,
    )


class BinaryHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[labels == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compute_binary_metrics(labels: np.ndarray, logits: np.ndarray, threshold: float) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    auc = binary_auc(labels, probs)

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc': auc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def make_pos_weight(labels: np.ndarray, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == 'none':
        return None
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0:
        return None
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)


def smooth_targets(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return labels
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    phase: str = 'train',
    epoch: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
    grad_clip: float = 0.0,
    target_smoothing: float = 0.0,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    all_logits = []
    all_labels = []

    desc = phase
    if epoch is not None and total_epochs is not None:
        desc = f'{phase} {epoch}/{total_epochs}'

    progress = None
    iterable = loader
    if show_progress and tqdm is not None:
        progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        iterable = progress

    for x, y in iterable:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss_y = smooth_targets(y, target_smoothing) if is_train else y

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, loss_y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

        if progress is not None:
            logits_np = np.concatenate(all_logits)
            labels_np = np.concatenate(all_labels)
            metrics = compute_binary_metrics(labels_np, logits_np, threshold)
            postfix = {
                'loss': f'{total_loss / max(total_count, 1):.4f}',
                'acc': f'{metrics["acc"]:.4f}',
                'f1': f'{metrics["f1"]:.4f}',
            }
            if optimizer is not None:
                postfix['lr'] = f'{optimizer.param_groups[0]["lr"]:.2e}'
            progress.set_postfix(postfix)

    if progress is not None:
        progress.close()

    logits_np = np.concatenate(all_logits) if all_logits else np.asarray([], dtype=np.float32)
    labels_np = np.concatenate(all_labels) if all_labels else np.asarray([], dtype=np.int64)
    metrics = compute_binary_metrics(labels_np, logits_np, threshold)
    return total_loss / max(total_count, 1), metrics


@torch.no_grad()
def evaluate_by_domain(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    domains: np.ndarray,
    batch_size: int,
    device: torch.device,
    threshold: float,
    show_progress: bool,
) -> dict[str, dict[str, float]]:
    model.eval()
    results = {}
    criterion = nn.BCEWithLogitsLoss()

    for domain in sorted(set(domains.tolist())):
        mask = domains == domain
        dataset = BinaryEmbeddingDataset(x[mask], y[mask])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        loss, metrics = run_epoch(
            model,
            loader,
            criterion,
            device,
            threshold,
            optimizer=None,
            phase=f'eval/{domain}',
            show_progress=show_progress,
        )
        results[domain] = {'loss': loss, **metrics}
        print_metrics(f'eval/{domain}', loss, metrics, n=int(mask.sum()))
    return results


def print_metrics(prefix: str, loss: float, metrics: dict[str, float], n: int | None = None) -> None:
    n_text = f', n={n:,}' if n is not None else ''
    auc = metrics['auc']
    auc_text = 'nan' if math.isnan(auc) else f'{auc:.4f}'
    print(
        f'{prefix}: loss={loss:.4f}, acc={metrics["acc"]:.4f}, '
        f'precision={metrics["precision"]:.4f}, recall={metrics["recall"]:.4f}, '
        f'f1={metrics["f1"]:.4f}, specificity={metrics["specificity"]:.4f}, '
        f'auc={auc_text}, tp={metrics["tp"]}, fp={metrics["fp"]}, '
        f'tn={metrics["tn"]}, fn={metrics["fn"]}{n_text}',
        flush=True,
    )


def is_improvement(value: float, best: float | None, monitor: str, min_delta: float) -> bool:
    if math.isnan(value):
        return False
    if best is None:
        return True
    if monitor == 'val_loss':
        return value < best - min_delta
    return value > best + min_delta


def monitor_value(loss: float, metrics: dict[str, float], monitor: str) -> float:
    if monitor == 'val_loss':
        return loss
    return float(metrics[monitor.removeprefix('val_')])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-domains', nargs='+', default=['PlantVillage'])
    parser.add_argument('--eval-domains', nargs='+', default=['PlantDoc', 'PlantWild'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--hidden-dim', type=int, default=0, help='0 means linear binary head')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pos-weight', choices=['none', 'balanced'], default='balanced')
    parser.add_argument('--target-smoothing', type=float, default=0.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--grad-clip', type=float, default=0.0, help='0 disables gradient clipping')
    parser.add_argument('--monitor', choices=['val_loss', 'val_f1', 'val_auc', 'val_recall'], default='val_loss')
    parser.add_argument('--early-stop-patience', type=int, default=10, help='<=0 disables early stopping')
    parser.add_argument('--early-stop-min-delta', type=float, default=1e-4)
    parser.add_argument('--lr-scheduler', choices=['none', 'plateau'], default='plateau')
    parser.add_argument('--lr-patience', type=int, default=3)
    parser.add_argument('--lr-factor', type=float, default=0.5)
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', default='runs/binary_disease_classifier.pt')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'device={device}', flush=True)

    show_progress = not args.no_progress
    if show_progress and tqdm is None:
        print('tqdm is not installed; progress bars are disabled.', flush=True)
        show_progress = False

    train_x, train_y, train_domains, skipped_train = concat_domains(args.train_domains)
    print(f'train={train_x.shape[0]:,}, skipped_train={skipped_train:,}', flush=True)

    train_idx, val_idx = split_train_val(train_y, args.val_ratio, args.seed)
    print(f'train_split={len(train_idx):,}, val_split={len(val_idx):,}', flush=True)

    train_loader = DataLoader(
        BinaryEmbeddingDataset(train_x[train_idx], train_y[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        BinaryEmbeddingDataset(train_x[val_idx], train_y[val_idx]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    model = BinaryHead(train_x.shape[1], args.hidden_dim, args.dropout).to(device)
    pos_weight = make_pos_weight(train_y[train_idx], args.pos_weight, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min' if args.monitor == 'val_loss' else 'max',
            factor=args.lr_factor,
            patience=args.lr_patience,
        )

    best_metric = None
    best_epoch = 0
    best_state = None
    best_val_loss = None
    best_val_metrics = None
    epochs_without_improvement = 0
    early_stopped = False
    started_at = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.threshold,
            optimizer,
            phase='train',
            epoch=epoch,
            total_epochs=args.epochs,
            show_progress=show_progress,
            grad_clip=args.grad_clip,
            target_smoothing=args.target_smoothing,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            eval_criterion,
            device,
            args.threshold,
            optimizer=None,
            phase='val',
            epoch=epoch,
            total_epochs=args.epochs,
            show_progress=show_progress,
        )

        current_metric = monitor_value(val_loss, val_metrics, args.monitor)
        improved = is_improvement(current_metric, best_metric, args.monitor, args.early_stop_min_delta)
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_val_loss = val_loss
            best_val_metrics = dict(val_metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(current_metric)

        elapsed = time.monotonic() - started_at
        eta = (elapsed / epoch) * (args.epochs - epoch)
        status = 'best' if improved else f'no_improve={epochs_without_improvement}'
        if args.early_stop_patience > 0:
            status += f'/{args.early_stop_patience}'

        print_metrics('train', train_loss, train_metrics)
        print_metrics('val', val_loss, val_metrics)
        print(
            f'epoch={epoch:03d}/{args.epochs:03d} '
            f'lr={optimizer.param_groups[0]["lr"]:.2e} '
            f'monitor={args.monitor}:{current_metric:.4f} status={status} '
            f'elapsed={format_duration(elapsed)} eta={format_duration(eta)}',
            flush=True,
        )

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            early_stopped = True
            print(
                f'early stopping at epoch={epoch}; best_epoch={best_epoch}, '
                f'best_{args.monitor}={best_metric:.4f}',
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if args.eval_domains:
        eval_x, eval_y, eval_domains, skipped_eval = concat_domains(args.eval_domains)
        print(f'eval={eval_x.shape[0]:,}, skipped_eval={skipped_eval:,}', flush=True)
        eval_results = evaluate_by_domain(
            model,
            eval_x,
            eval_y,
            eval_domains,
            args.batch_size,
            device,
            args.threshold,
            show_progress,
        )
    else:
        eval_results = {}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state': model.state_dict(),
            'args': vars(args),
            'target_names': ['healthy', 'diseased'],
            'threshold': args.threshold,
            'best_epoch': best_epoch,
            'best_metric': best_metric,
            'best_val_loss': best_val_loss,
            'best_val_metrics': best_val_metrics,
            'early_stopped': early_stopped,
            'eval_results': eval_results,
        },
        output_path,
    )
    print(f'saved={output_path}', flush=True)
    print(
        json.dumps(
            {
                'best_epoch': best_epoch,
                'best_metric': best_metric,
                'best_val_loss': best_val_loss,
                'best_val_metrics': best_val_metrics,
                'early_stopped': early_stopped,
                'eval_results': eval_results,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
