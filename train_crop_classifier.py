#!/usr/bin/env python3
"""Train a lightweight classifier on cached DINOv3 embeddings.

Recommended first experiment:
    python train_crop_classifier.py \
        --train-domains PlantVillage \
        --eval-domains PlantDoc PlantWild \
        --label-mode crop
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


ROOT = Path('/root/DINO3')
CACHE_FILES = {
    'PlantVillage': ROOT / 'cache/dinov3_plantvillage_color_embeddings.pt',
    'PlantDoc': ROOT / 'cache/dinov3_plantdoc_embeddings.pt',
    'PlantWild': ROOT / 'cache/dinov3_plantwild_embeddings.pt',
}

PLANTVILLAGE_ROOT = ROOT / 'dataset/PlantVillage'
PLANTDOC_ROOT = ROOT / 'dataset/PlantDoc-Object-Detection-Dataset'
PLANTWILD_ROOT = ROOT / 'dataset/plantwild/images'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.from_numpy(embeddings.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.labels[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cached_embeddings(cache_path: Path) -> np.ndarray:
    if not cache_path.exists():
        raise FileNotFoundError(f'Cache not found: {cache_path}')

    try:
        payload = torch.load(cache_path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location='cpu')

    if not isinstance(payload, dict) or 'embeddings' not in payload:
        raise ValueError(f'Invalid cache format: {cache_path}')

    emb = payload['embeddings']
    if not torch.is_tensor(emb):
        raise ValueError(f'`embeddings` is not a torch tensor: {cache_path}')

    return emb.detach().cpu().numpy().astype(np.float32)


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def load_plantvillage_samples() -> tuple[list[Path], list[str]]:
    split_files = [
        PLANTVILLAGE_ROOT / 'splits/color_train.txt',
        PLANTVILLAGE_ROOT / 'splits/color_test.txt',
    ]

    image_paths = []
    for split_file in split_files:
        if not split_file.exists():
            continue
        for line in split_file.read_text(encoding='utf-8').splitlines():
            rel = line.strip()
            if not rel:
                continue
            path = PLANTVILLAGE_ROOT / rel
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                image_paths.append(path)

    if not image_paths:
        color_root = PLANTVILLAGE_ROOT / 'raw/color'
        image_paths = [
            path for path in color_root.rglob('*')
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]

    image_paths = sorted(set(image_paths))
    class_labels = [path.parent.name for path in image_paths]
    return image_paths, class_labels


def collect_plantdoc_split_samples(split_name: str, csv_path: Path) -> tuple[list[Path], list[str]]:
    split_dir = PLANTDOC_ROOT / split_name
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    df = pd.read_csv(csv_path)
    grouped = df.groupby('filename', sort=False)['class'].apply(
        lambda series: unique_preserve_order([
            str(value).strip()
            for value in series
            if pd.notna(value) and str(value).strip()
        ])
    )

    image_paths = []
    class_labels = []
    for filename, classes in grouped.items():
        path = split_dir / str(filename)
        if not path.exists() or path.suffix.lower() not in IMAGE_EXTS or not classes:
            continue
        image_paths.append(path)
        class_labels.append(' | '.join(classes))

    return image_paths, class_labels


def load_plantdoc_samples() -> tuple[list[Path], list[str]]:
    train_paths, train_labels = collect_plantdoc_split_samples(
        'TRAIN', PLANTDOC_ROOT / 'train_labels.csv'
    )
    test_paths, test_labels = collect_plantdoc_split_samples(
        'TEST', PLANTDOC_ROOT / 'test_labels.csv'
    )
    return train_paths + test_paths, train_labels + test_labels


def load_plantwild_samples() -> tuple[list[Path], list[str]]:
    image_paths = [
        path for path in PLANTWILD_ROOT.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    image_paths = sorted(image_paths)
    class_labels = [path.parent.name for path in image_paths]
    return image_paths, class_labels


def load_domain_samples(domain: str) -> tuple[list[Path], list[str]]:
    if domain == 'PlantVillage':
        return load_plantvillage_samples()
    if domain == 'PlantDoc':
        return load_plantdoc_samples()
    if domain == 'PlantWild':
        return load_plantwild_samples()
    raise ValueError(f'Unsupported domain: {domain}')


def extract_crop_name(domain: str, class_label: str) -> str:
    label = str(class_label).strip()

    if domain == 'PlantVillage':
        crop_raw = label.split('___')[0].lower().strip()
        crop_raw = crop_raw.replace('_', ' ').replace(',', ' ')
        crop_raw = crop_raw.replace('(', ' ').replace(')', ' ')
        crop_raw = ' '.join(crop_raw.split())
        alias = {
            'pepper bell': 'bell pepper',
            'cherry including sour': 'cherry',
            'corn maize': 'corn',
        }
        return alias.get(crop_raw, crop_raw)

    if domain == 'PlantDoc':
        first = label.split('|')[0].strip().lower()
        if 'bell_pepper' in first or 'bell pepper' in first:
            return 'bell pepper'
        for crop in [
            'apple', 'blueberry', 'cherry', 'corn', 'grape', 'peach', 'potato',
            'raspberry', 'squash', 'strawberry', 'tomato',
        ]:
            if first.startswith(crop):
                return crop
        if first.startswith('soyabean') or first.startswith('soybean'):
            return 'soybean'
        return first.split()[0] if first else 'unknown'

    words = label.lower().split()
    if len(words) >= 2 and f'{words[0]} {words[1]}' == 'bell pepper':
        crop = 'bell pepper'
    elif words:
        crop = words[0]
    else:
        crop = 'unknown'
    return 'grape' if crop == 'grapevine' else crop


def make_labels(domain: str, class_labels: list[str], label_mode: str) -> list[str]:
    if label_mode == 'class':
        return [str(label) for label in class_labels]
    if label_mode == 'crop':
        return [extract_crop_name(domain, label) for label in class_labels]
    raise ValueError(f'Unsupported label mode: {label_mode}')


def load_domain_arrays(domain: str, label_mode: str) -> tuple[np.ndarray, np.ndarray]:
    embeddings = load_cached_embeddings(CACHE_FILES[domain])
    _, class_labels = load_domain_samples(domain)
    if len(class_labels) != embeddings.shape[0]:
        raise ValueError(
            f'[{domain}] label count and embedding count mismatch: '
            f'labels={len(class_labels):,}, embeddings={embeddings.shape[0]:,}'
        )
    return embeddings, np.asarray(make_labels(domain, class_labels, label_mode), dtype=object)


def read_label_filter(path: str | None) -> set[str] | None:
    if not path:
        return None
    labels = [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines()]
    return {label for label in labels if label}


def build_vocab(labels: np.ndarray, label_filter: set[str] | None) -> list[str]:
    label_set = sorted(set(labels.tolist()))
    if label_filter is not None:
        label_set = [label for label in label_set if label in label_filter]
    if not label_set:
        raise ValueError('No labels left after applying label filter.')
    return label_set


def encode_with_vocab(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    keep = np.asarray([label in label_to_id for label in labels], dtype=bool)
    encoded = np.asarray([label_to_id[label] for label in labels[keep]], dtype=np.int64)
    return embeddings[keep], encoded, int((~keep).sum())


def concat_domains(
    domains: list[str],
    label_mode: str,
    label_to_id: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    all_x = []
    all_y = []
    all_domains = []
    skipped = 0

    for domain in domains:
        embeddings, labels = load_domain_arrays(domain, label_mode)
        if label_to_id is None:
            y = labels
            x = embeddings
            skipped_domain = 0
        else:
            x, y, skipped_domain = encode_with_vocab(embeddings, labels, label_to_id)
        all_x.append(x)
        all_y.append(y)
        all_domains.extend([domain] * x.shape[0])
        skipped += skipped_domain
        print(f'{domain}: used={x.shape[0]:,}, skipped={skipped_domain:,}')

    return (
        np.concatenate(all_x, axis=0),
        np.concatenate(all_y, axis=0),
        np.asarray(all_domains, dtype=object),
        skipped,
    )


def split_train_val(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError('--val-ratio must be between 0 and 1.')

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for label in sorted(set(labels.tolist())):
        cls_indices = np.flatnonzero(labels == label)
        rng.shuffle(cls_indices)
        if cls_indices.shape[0] == 1:
            train_indices.extend(cls_indices.tolist())
            continue

        val_size = max(1, int(round(cls_indices.shape[0] * val_ratio)))
        val_size = min(val_size, cls_indices.shape[0] - 1)
        val_indices.extend(cls_indices[:val_size].tolist())
        train_indices.extend(cls_indices[val_size:].tolist())

    if not train_indices or not val_indices:
        raise ValueError('Could not create non-empty train/val split. Check val ratio and labels.')

    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_model(input_dim: int, num_classes: int, hidden_dim: int, dropout: float) -> nn.Module:
    if hidden_dim <= 0:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes),
        )
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    phase: str = 'train',
    epoch: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
    grad_clip: float = 0.0,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

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

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_count += batch_size

        if progress is not None:
            running_loss = total_loss / max(total_count, 1)
            running_acc = total_correct / max(total_count, 1)
            postfix = {'loss': f'{running_loss:.4f}', 'acc': f'{running_acc:.4f}'}
            if optimizer is not None:
                postfix['lr'] = f'{optimizer.param_groups[0]["lr"]:.2e}'
            progress.set_postfix(postfix)

    if progress is not None:
        progress.close()

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def evaluate_by_domain(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    domains: np.ndarray,
    batch_size: int,
    device: torch.device,
    show_progress: bool,
) -> dict[str, float]:
    model.eval()
    results = {}
    for domain in sorted(set(domains.tolist())):
        mask = domains == domain
        dataset = EmbeddingDataset(x[mask], y[mask])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        loss, acc = run_epoch(
            model,
            loader,
            criterion,
            device,
            optimizer=None,
            phase=f'eval/{domain}',
            show_progress=show_progress,
        )
        results[domain] = acc
        print(f'eval/{domain}: loss={loss:.4f}, acc={acc:.4f}, n={mask.sum():,}', flush=True)
    return results


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.clip(counts, 1.0, None))
    return torch.from_numpy(weights.astype(np.float32))


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f'{hours:d}h {minutes:02d}m {seconds:02d}s'
    if minutes:
        return f'{minutes:d}m {seconds:02d}s'
    return f'{seconds:d}s'


def is_improvement(value: float, best: float | None, monitor: str, min_delta: float) -> bool:
    if best is None:
        return True
    if monitor == 'val_loss':
        return value < best - min_delta
    return value > best + min_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-domains', nargs='+', default=['PlantVillage'])
    parser.add_argument('--eval-domains', nargs='+', default=['PlantDoc'])
    parser.add_argument('--label-mode', choices=['crop', 'class'], default='crop')
    parser.add_argument('--label-filter-file', default=None, help='Optional newline-separated label allowlist, e.g. class')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--hidden-dim', type=int, default=0, help='0 means linear probe')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--class-weights', choices=['none', 'balanced'], default='none')
    parser.add_argument('--grad-clip', type=float, default=0.0, help='0 disables gradient clipping')
    parser.add_argument('--monitor', choices=['val_loss', 'val_acc'], default='val_loss')
    parser.add_argument('--early-stop-patience', type=int, default=10, help='<=0 disables early stopping')
    parser.add_argument('--early-stop-min-delta', type=float, default=1e-4)
    parser.add_argument('--lr-scheduler', choices=['none', 'plateau'], default='plateau')
    parser.add_argument('--lr-patience', type=int, default=3)
    parser.add_argument('--lr-factor', type=float, default=0.5)
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', default='runs/embedding_classifier.pt')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'device={device}')

    train_x_raw, train_labels_raw, _, _ = concat_domains(
        args.train_domains,
        args.label_mode,
        label_to_id=None,
    )
    label_filter = read_label_filter(args.label_filter_file)
    label_names = build_vocab(train_labels_raw, label_filter)
    label_to_id = {label: idx for idx, label in enumerate(label_names)}

    train_x, train_y, train_domains, skipped_train = concat_domains(
        args.train_domains,
        args.label_mode,
        label_to_id=label_to_id,
    )
    print(f'labels={len(label_names)}, train={train_x.shape[0]:,}, skipped_train={skipped_train:,}')

    show_progress = not args.no_progress
    if show_progress and tqdm is None:
        print('tqdm is not installed; progress bars are disabled.', flush=True)
        show_progress = False

    train_idx, val_idx = split_train_val(train_y, args.val_ratio, args.seed)
    print(f'train_split={len(train_idx):,}, val_split={len(val_idx):,}', flush=True)
    train_loader = DataLoader(
        EmbeddingDataset(train_x[train_idx], train_y[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        EmbeddingDataset(train_x[val_idx], train_y[val_idx]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    model = build_model(train_x.shape[1], len(label_names), args.hidden_dim, args.dropout).to(device)
    class_weight = None
    if args.class_weights == 'balanced':
        class_weight = compute_class_weights(train_y[train_idx], len(label_names)).to(device)
    train_criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
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
    best_val_loss = None
    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0
    early_stopped = False
    started_at = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            train_criterion,
            device,
            optimizer,
            phase='train',
            epoch=epoch,
            total_epochs=args.epochs,
            show_progress=show_progress,
            grad_clip=args.grad_clip,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            eval_criterion,
            device,
            optimizer=None,
            phase='val',
            epoch=epoch,
            total_epochs=args.epochs,
            show_progress=show_progress,
        )

        monitor_value = val_loss if args.monitor == 'val_loss' else val_acc
        improved = is_improvement(monitor_value, best_metric, args.monitor, args.early_stop_min_delta)
        if improved:
            best_metric = monitor_value
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(monitor_value)

        elapsed = time.monotonic() - started_at
        avg_epoch_time = elapsed / epoch
        eta = avg_epoch_time * (args.epochs - epoch)
        status = 'best' if improved else f'no_improve={epochs_without_improvement}'
        if args.early_stop_patience > 0:
            status += f'/{args.early_stop_patience}'
        print(
            f'epoch={epoch:03d}/{args.epochs:03d} '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} '
            f'lr={optimizer.param_groups[0]["lr"]:.2e} '
            f'monitor={args.monitor}:{monitor_value:.4f} status={status} '
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
        eval_x, eval_y, eval_domains, skipped_eval = concat_domains(
            args.eval_domains,
            args.label_mode,
            label_to_id=label_to_id,
        )
        print(f'eval={eval_x.shape[0]:,}, skipped_eval={skipped_eval:,}', flush=True)
        eval_results = evaluate_by_domain(
            model,
            eval_x,
            eval_y,
            eval_domains,
            args.batch_size,
            device,
            show_progress,
        )
    else:
        eval_results = {}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state': model.state_dict(),
            'label_names': label_names,
            'args': vars(args),
            'best_epoch': best_epoch,
            'best_metric': best_metric,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
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
                'best_val_acc': best_val_acc,
                'early_stopped': early_stopped,
                'eval_results': eval_results,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
