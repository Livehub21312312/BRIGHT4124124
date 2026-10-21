#!/usr/bin/env python3
"""Run a short, real-data UNet training smoke test on the historical 9 events.

This is an independent Windows-compatible wrapper around the repository's
unmodified Dataset, UNet, and Lovasz implementations.  It validates the three
official split files, selects only a small training subset, and checks that the
official loss can run forward, backward, and update parameters.  It never
reads validation/test TIFFs and it is not a full training entry point.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from tools.smoke_test_real_data import (  # noqa: E402
    BASELINE_ROOT,
    PAPER_EVENTS,
    MultimodalDamageAssessmentDatset,
    UNet,
    event_for_sample,
    triplet_paths,
)
from util_func import lovasz_loss as L  # noqa: E402


SPLIT_ROOT = BASELINE_ROOT / "dataset" / "splitname" / "standard_ML"
EXPECTED_PAPER9_COUNTS = {"train": 2229, "val": 349, "test": 661}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a few real-data UNet optimization steps using the official "
            "BRIGHT data loader and loss implementation."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="BRIGHT root containing pre-event, post-event, and target.",
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=SPLIT_ROOT / "train_set.txt",
        help="Official standard_ML training list.",
    )
    parser.add_argument(
        "--val-list",
        type=Path,
        default=SPLIT_ROOT / "val_set.txt",
        help="Official standard_ML validation list (IDs only; TIFFs are not read).",
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        default=SPLIT_ROOT / "test_set.txt",
        help="Official standard_ML test list (IDs only; TIFFs are not read).",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision. This differs from official FP32.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "paper9_train_smoke"
        / "training_smoke_report.json",
    )
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split list does not exist: {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [sample_id for sample_id in ids if sample_id]
    duplicates = sorted(
        sample_id for sample_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate IDs in {path}: {duplicates[:10]}")
    return ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_paper9(ids: list[str]) -> list[str]:
    return [sample_id for sample_id in ids if event_for_sample(sample_id)]


def validate_splits(args: argparse.Namespace) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for name, path in (
        ("train", args.train_list),
        ("val", args.val_list),
        ("test", args.test_list),
    ):
        paper9_ids = select_paper9(read_ids(path))
        expected = EXPECTED_PAPER9_COUNTS[name]
        if len(paper9_ids) != expected:
            raise ValueError(
                f"Expected {expected} paper-nine {name} IDs, got {len(paper9_ids)}"
            )
        selected[name] = paper9_ids

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(set(selected[left]) & set(selected[right]))
        if overlap:
            raise ValueError(f"{left}/{right} overlap: {overlap[:10]}")
    return selected


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    for name in ("steps", "sample_limit", "micro_batch_size", "crop_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp is only supported with a CUDA device")


def finite_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite loss encountered: {value}")
    return value


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    validate_args(args, device)
    set_seed(args.seed)

    splits = validate_splits(args)
    smoke_ids = splits["train"][: args.sample_limit]
    missing = [
        str(path)
        for sample_id in smoke_ids
        for path in triplet_paths(args.dataset_root, sample_id).values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Smoke subset is missing TIFF files:\n" + "\n".join(missing[:20])
        )

    dataset = MultimodalDamageAssessmentDatset(
        str(args.dataset_root),
        smoke_ids,
        crop_size=args.crop_size,
        max_iters=None,
        type="train",
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )
    if args.steps > len(loader):
        raise ValueError(
            f"--steps {args.steps} exceeds the {len(loader)} available smoke batches; "
            "increase --sample-limit or reduce --steps"
        )

    model = UNet(in_channels=6, num_classes=4).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    tracked_parameter = next(model.parameters())
    initial_parameter = tracked_parameter.detach().cpu().clone()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    step_reports: list[dict[str, Any]] = []

    for step, batch in enumerate(loader, start=1):
        if step > args.steps:
            break
        pre, post, _labels_loc, labels_clf, sample_ids = batch
        inputs = torch.cat([pre, post], dim=1).to(device)
        labels_clf = labels_clf.to(device).long()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(inputs)
            ce_loss = F.cross_entropy(logits, labels_clf)
            lovasz_loss = L.lovasz_softmax(
                F.softmax(logits, dim=1), labels_clf, ignore=255
            )
            total_loss = ce_loss + 0.75 * lovasz_loss

        finite_float(total_loss.detach().item())
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step_reports.append(
            {
                "step": step,
                "sample_ids": list(sample_ids),
                "input_shape": list(inputs.shape),
                "output_shape": list(logits.shape),
                "ce_loss": finite_float(ce_loss.detach().item()),
                "lovasz_loss": finite_float(lovasz_loss.detach().item()),
                "total_loss_ce_plus_0_75_lovasz": finite_float(
                    total_loss.detach().item()
                ),
            }
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    parameter_max_abs_change = float(
        (tracked_parameter.detach().cpu() - initial_parameter).abs().max().item()
    )
    if not math.isfinite(parameter_max_abs_change) or parameter_max_abs_change == 0:
        raise RuntimeError("The tracked model parameter did not change")

    cuda_memory = None
    if device.type == "cuda":
        cuda_memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }

    report: dict[str, Any] = {
        "status": "passed",
        "scope": (
            "real-data training smoke test only; not a paper result and not a "
            "completed training run"
        ),
        "official_sources_reused_unmodified": [
            "bda_benchmark/dataset/make_data_loader.py",
            "bda_benchmark/dataset/imutils.py",
            "bda_benchmark/model/UNet.py",
            "bda_benchmark/util_func/lovasz_loss.py",
        ],
        "split": {
            "selection": "historical nine-event subset of current standard_ML lists",
            "events": list(PAPER_EVENTS),
            "verified_counts": {name: len(ids) for name, ids in splits.items()},
            "source_lists": {
                name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in (
                    ("train", args.train_list),
                    ("val", args.val_list),
                    ("test", args.test_list),
                )
            },
            "overlap_count": 0,
            "training_triplets_existence_checked": len(smoke_ids),
            "training_samples_loaded": sum(
                len(step_report["sample_ids"]) for step_report in step_reports
            ),
            "validation_tiffs_read": 0,
            "test_tiffs_read": 0,
        },
        "configuration": {
            "dataset_root": str(args.dataset_root.resolve()),
            "steps": args.steps,
            "sample_limit": args.sample_limit,
            "micro_batch_size": args.micro_batch_size,
            "crop_size": args.crop_size,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss": "cross_entropy + 0.75 * lovasz_softmax",
            "seed": args.seed,
            "amp": use_amp,
            "num_workers": 0,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "elapsed_seconds_for_optimizer_steps": elapsed,
            "seconds_per_optimizer_step": elapsed / args.steps,
            "cuda_memory": cuda_memory,
        },
        "parameter_update": {
            "tracked_parameter": next(iter(model.state_dict().keys())),
            "max_absolute_change": parameter_max_abs_change,
            "changed": True,
        },
        "steps": step_reports,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report written to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
