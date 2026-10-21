#!/usr/bin/env python3
"""Smoke-test UNet training on the 14-event official standard_ML train split.

The script imports the repository's unmodified Dataset, augmentation, UNet,
and Lovasz implementations.  It reads only the official training and
validation ID lists, loads a small subset of training TIFFs, and performs a few
optimizer updates.  No test list is accepted or opened, and no validation TIFF
is read.  This is a training-pipeline check, not a completed experiment.
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
    MultimodalDamageAssessmentDatset,
    UNet,
    triplet_paths,
)
from util_func import lovasz_loss as L  # noqa: E402


SPLIT_ROOT = BASELINE_ROOT / "dataset" / "splitname" / "standard_ML"
TRAIN_LIST = SPLIT_ROOT / "train_set.txt"
VAL_LIST = SPLIT_ROOT / "val_set.txt"
EXPECTED_COUNTS = {"train": 2918, "val": 450}
EXPECTED_EVENTS = {
    "bata-explosion",
    "beirut-explosion",
    "congo-volcano",
    "haiti-earthquake",
    "hawaii-wildfire",
    "la_palma-volcano",
    "libya-flood",
    "marshall-wildfire",
    "mexico-hurricane",
    "morocco-earthquake",
    "myanmar-hurricane",
    "noto-earthquake",
    "turkey-earthquake",
    "ukraine-conflict",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a few real-data UNet optimizer steps using all-event official "
            "standard_ML training IDs."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="BRIGHT root containing pre-event, post-event, and target.",
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
        help="Use CUDA automatic mixed precision; this differs from official FP32.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "standard_ml_train_smoke"
        / "training_smoke_report.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_for_sample(sample_id: str) -> str:
    if "_" not in sample_id:
        raise ValueError(f"Sample ID has no numeric suffix: {sample_id}")
    return sample_id.rsplit("_", 1)[0]


def read_unique_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Official split list does not exist: {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [sample_id for sample_id in ids if sample_id]
    duplicates = sorted(
        sample_id for sample_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate IDs in {path}: {duplicates[:10]}")
    return ids


def validate_official_train_val_lists() -> dict[str, list[str]]:
    splits = {"train": read_unique_ids(TRAIN_LIST), "val": read_unique_ids(VAL_LIST)}
    for name, ids in splits.items():
        if len(ids) != EXPECTED_COUNTS[name]:
            raise ValueError(
                f"Expected {EXPECTED_COUNTS[name]} official {name} IDs, got {len(ids)}"
            )
        events = {event_for_sample(sample_id) for sample_id in ids}
        if events != EXPECTED_EVENTS:
            raise ValueError(
                f"Official {name} events differ: missing={sorted(EXPECTED_EVENTS-events)}, "
                f"unexpected={sorted(events-EXPECTED_EVENTS)}"
            )

    overlap = sorted(set(splits["train"]) & set(splits["val"]))
    if overlap:
        raise ValueError(f"Official train/val overlap: {overlap[:10]}")
    return splits


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


def event_counts(ids: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(event_for_sample(sample_id) for sample_id in ids).items()))


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    validate_args(args, device)
    set_seed(args.seed)

    splits = validate_official_train_val_lists()
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
    tracked_parameter_name, tracked_parameter = next(iter(model.named_parameters()))
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
        expected_input_shape = (
            len(sample_ids),
            6,
            args.crop_size,
            args.crop_size,
        )
        if tuple(inputs.shape) != expected_input_shape:
            raise ValueError(
                f"Unexpected input shape {tuple(inputs.shape)}; "
                f"expected {expected_input_shape}"
            )
        label_values = sorted(int(value) for value in torch.unique(labels_clf).tolist())
        invalid_labels = sorted(set(label_values) - {0, 1, 2, 3})
        if invalid_labels:
            raise ValueError(f"Training crop contains invalid labels: {invalid_labels}")
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(inputs)
            expected_output_shape = (
                len(sample_ids),
                4,
                args.crop_size,
                args.crop_size,
            )
            if tuple(logits.shape) != expected_output_shape:
                raise ValueError(
                    f"Unexpected output shape {tuple(logits.shape)}; "
                    f"expected {expected_output_shape}"
                )
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Model output contains NaN or infinity")
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
                "events": [event_for_sample(sample_id) for sample_id in sample_ids],
                "input_shape": list(inputs.shape),
                "output_shape": list(logits.shape),
                "label_values": label_values,
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
        "experiment_role": "essd_2025_standard_ml_14_event_main_experiment",
        "scope": (
            "real-data training smoke test only; not a paper metric and not a "
            "completed training run"
        ),
        "official_sources_reused_unmodified": [
            "bda_benchmark/dataset/make_data_loader.py",
            "bda_benchmark/dataset/imutils.py",
            "bda_benchmark/model/UNet.py",
            "bda_benchmark/util_func/lovasz_loss.py",
        ],
        "data_access": {
            "training_list": {
                "path": str(TRAIN_LIST.resolve()),
                "sha256": sha256_file(TRAIN_LIST),
                "count": len(splits["train"]),
                "event_counts": event_counts(splits["train"]),
            },
            "validation_list_ids_only": {
                "path": str(VAL_LIST.resolve()),
                "sha256": sha256_file(VAL_LIST),
                "count": len(splits["val"]),
                "event_counts": event_counts(splits["val"]),
            },
            "train_val_overlap_count": 0,
            "training_triplets_existence_checked": len(smoke_ids),
            "training_samples_loaded": sum(
                len(step_report["sample_ids"]) for step_report in step_reports
            ),
            "validation_tiffs_read": 0,
            "test_list_accessed": False,
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
            "tracked_parameter": tracked_parameter_name,
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
