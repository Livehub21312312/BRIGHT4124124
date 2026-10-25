#!/usr/bin/env python3
"""Train the official BRIGHT UNet on the 14-event standard_ML split.

The default 4 GB GPU configuration uses micro-batch 1 with 16-step gradient
accumulation and AMP. It reads only official train/val manifests. Test
evaluation is intentionally delegated to the separate evaluation command after
training has finished.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "bda_benchmark"
CACHE_ROOT = Path(tempfile.gettempdir()) / "bright-reproduction-cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))
sys.dont_write_bytecode = True

from dataset.make_data_loader import MultimodalDamageAssessmentDatset  # noqa: E402
from model.UNet import UNet  # noqa: E402
from util_func.lovasz_loss import lovasz_softmax  # noqa: E402
from util_func.metrics import Evaluator  # noqa: E402


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
TOTAL_SAMPLE_EXPOSURES = 800_000
CHECKPOINT_VERSION = 1


class ResumableRandomSampler(Sampler[int]):
    """One deterministic shuffled pass with a resumable sample cursor."""

    def __init__(self, length: int, seed: int, start_index: int = 0) -> None:
        if not 0 <= start_index <= length:
            raise ValueError(f"Invalid sampler cursor {start_index}/{length}")
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.order = torch.randperm(length, generator=generator).tolist()
        self.start_index = start_index

    def __iter__(self) -> Iterator[int]:
        yield from self.order[self.start_index :]

    def __len__(self) -> int:
        return len(self.order) - self.start_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train official BRIGHT UNet on standard_ML train/val only."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "standard_ml_unet_formal",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from latest.pt or best_training_state.pt in a prior run.",
    )
    parser.add_argument("--sample-exposures", type=int, default=TOTAL_SAMPLE_EXPOSURES)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--validation-every-updates", type=int, default=500)
    parser.add_argument("--checkpoint-every-updates", type=int, default=500)
    parser.add_argument("--max-consecutive-amp-skips", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA AMP (default); use --no-amp for full FP32.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("".join(f"{line}\n" for line in lines).encode()).hexdigest()


def event_for_sample(sample_id: str) -> str:
    if "_" not in sample_id:
        raise ValueError(f"Sample ID has no numeric suffix: {sample_id}")
    return sample_id.rsplit("_", 1)[0]


def read_ids(path: Path, label: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Official {label} list missing: {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [sample_id for sample_id in ids if sample_id]
    duplicates = sorted(
        sample_id for sample_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate IDs in {label} list: {duplicates[:10]}")
    if len(ids) != EXPECTED_COUNTS[label]:
        raise ValueError(
            f"Expected {EXPECTED_COUNTS[label]} {label} IDs, got {len(ids)}"
        )
    events = {event_for_sample(sample_id) for sample_id in ids}
    if events != EXPECTED_EVENTS:
        raise ValueError(
            f"{label} event set differs: missing={sorted(EXPECTED_EVENTS-events)}, "
            f"unexpected={sorted(events-EXPECTED_EVENTS)}"
        )
    return ids


def read_splits() -> dict[str, list[str]]:
    splits = {
        "train": read_ids(TRAIN_LIST, "train"),
        "val": read_ids(VAL_LIST, "val"),
    }
    overlap = sorted(set(splits["train"]) & set(splits["val"]))
    if overlap:
        raise ValueError(f"Official train/val overlap: {overlap[:10]}")
    return splits


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
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
    positive = (
        "sample_exposures",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "crop_size",
        "validation_every_updates",
        "checkpoint_every_updates",
        "max_consecutive_amp_skips",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid learning rate/weight decay")
    if args.sample_exposures % args.micro_batch_size:
        raise ValueError(
            f"sample exposures {args.sample_exposures} must be divisible by "
            f"micro-batch size {args.micro_batch_size}; partial accumulation "
            "groups are supported, but partial data batches are not"
        )
    if args.amp and device.type != "cuda":
        raise ValueError("AMP requires CUDA; use --no-amp for CPU/FP32")
    class_weights = getattr(args, "class_weights", None)
    if class_weights is not None and (
        len(class_weights) != 4
        or not all(math.isfinite(float(value)) and float(value) > 0 for value in class_weights)
    ):
        raise ValueError("class_weights must contain four finite positive values")
    milestones = getattr(args, "validation_sample_exposures", [])
    if milestones != sorted(set(milestones)) or any(
        value <= 0 or value > args.sample_exposures for value in milestones
    ):
        raise ValueError("validation_sample_exposures must be unique increasing values within budget")


def critical_config(args: argparse.Namespace, amp_enabled: bool) -> dict[str, Any]:
    full_budget = args.sample_exposures == TOTAL_SAMPLE_EXPOSURES
    initial_checkpoint = getattr(args, "initial_checkpoint", None)
    return {
        "experiment_mode": getattr(
            args,
            "experiment_mode",
            "official_full_exposure_budget_resource_adapted"
            if full_budget
            else "reduced_budget_resource_adapted_not_full_paper_reproduction",
        ),
        "paper_reproduction_status": (
            "full_exposure_budget_but_resource_adapted_not_exact_reproduction"
            if full_budget
            else "reduced_exposure_budget_not_full_paper_reproduction"
        ),
        "dataset_root": str(args.dataset_root.resolve()),
        "requested_device": args.device,
        "sample_exposures": args.sample_exposures,
        "initial_checkpoint": (
            str(Path(initial_checkpoint).resolve()) if initial_checkpoint else None
        ),
        "initial_checkpoint_sha256": (
            sha256_file(Path(initial_checkpoint))
            if initial_checkpoint and Path(initial_checkpoint).is_file()
            else None
        ),
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.micro_batch_size
        * args.gradient_accumulation_steps,
        "planned_optimizer_update_attempts": math.ceil(
            args.sample_exposures
            / (args.micro_batch_size * args.gradient_accumulation_steps)
        ),
        "crop_size": args.crop_size,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": None,
        "loss": (
            "weighted_cross_entropy(class_weights) + 0.75 * official lovasz_softmax"
            if getattr(args, "class_weights", None) is not None
            else "cross_entropy + 0.75 * official lovasz_softmax"
        ),
        "class_weights": getattr(args, "class_weights", None),
        "class_weight_metadata": getattr(args, "class_weight_metadata", None),
        "amp": amp_enabled,
        "validation_every_successful_updates": args.validation_every_updates,
        "validation_sample_exposure_milestones": getattr(
            args, "validation_sample_exposures", []
        ),
        "checkpoint_every_update_attempts": args.checkpoint_every_updates,
        "max_consecutive_amp_skips": args.max_consecutive_amp_skips,
        "seed": args.seed,
        "num_workers": 0,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def next_failure_report_path(output_dir: Path) -> Path:
    base = output_dir / "failure_report.json"
    if not base.exists():
        return base
    index = 2
    while (output_dir / f"failure_report_{index}.json").exists():
        index += 1
    return output_dir / f"failure_report_{index}.json"


def torch_load_full(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def finite_number(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def metrics_from_evaluators(
    evaluator_loc: Evaluator,
    evaluator_clf: Evaluator,
    evaluator_total: Evaluator,
) -> dict[str, Any]:
    with np.errstate(divide="ignore", invalid="ignore"):
        loc_f1 = evaluator_loc.Pixel_F1_score()
        damage_f1 = evaluator_clf.Damage_F1_score()
        clf_f1 = len(damage_f1) / np.sum(1.0 / damage_f1)
        oa = evaluator_total.Pixel_Accuracy()
        class_iou = evaluator_total.Intersection_over_Union()
        miou = evaluator_total.Mean_Intersection_over_Union()
        tp, fn, fp = evaluator_clf.calculate_per_class_metrics()
        damage_precision = tp / (tp + fp + 1e-7)
        damage_recall = tp / (tp + fn + 1e-7)
    target_counts = evaluator_total.confusion_matrix.sum(axis=1)
    prediction_counts = evaluator_total.confusion_matrix.sum(axis=0)
    valid_pixels = int(target_counts.sum())
    return {
        "f1loc": finite_number(loc_f1),
        "f1clf_harmonic_mean": finite_number(clf_f1),
        "oa": finite_number(oa),
        "miou": finite_number(miou),
        "per_class_f1_damage_only": {
            "intact": finite_number(damage_f1[0]),
            "damaged": finite_number(damage_f1[1]),
            "destroyed": finite_number(damage_f1[2]),
        },
        "per_class_precision_damage_only": {
            "intact": finite_number(damage_precision[0]),
            "damaged": finite_number(damage_precision[1]),
            "destroyed": finite_number(damage_precision[2]),
        },
        "per_class_recall_damage_only": {
            "intact": finite_number(damage_recall[0]),
            "damaged": finite_number(damage_recall[1]),
            "destroyed": finite_number(damage_recall[2]),
        },
        "ground_truth_pixel_counts": {
            name: int(target_counts[index])
            for index, name in enumerate(("background", "intact", "damaged", "destroyed"))
        },
        "prediction_pixel_counts": {
            name: int(prediction_counts[index])
            for index, name in enumerate(("background", "intact", "damaged", "destroyed"))
        },
        "pixel_distribution_percent": {
            "ground_truth": {
                name: (float(target_counts[index] / valid_pixels * 100) if valid_pixels else None)
                for index, name in enumerate(("background", "intact", "damaged", "destroyed"))
            },
            "prediction": {
                name: (float(prediction_counts[index] / valid_pixels * 100) if valid_pixels else None)
                for index, name in enumerate(("background", "intact", "damaged", "destroyed"))
            },
        },
        "per_class_iou": {
            "background": finite_number(class_iou[0]),
            "intact": finite_number(class_iou[1]),
            "damaged": finite_number(class_iou[2]),
            "destroyed": finite_number(class_iou[3]),
        },
        "confusion_matrix_classification": evaluator_total.confusion_matrix.tolist(),
        "confusion_matrix_localization": evaluator_loc.confusion_matrix.tolist(),
        "confusion_matrix_damage_pixels": evaluator_clf.confusion_matrix.tolist(),
    }


def evaluate_validation(
    *,
    model: torch.nn.Module,
    dataset_root: Path,
    val_ids: list[str],
    crop_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    dataset = MultimodalDamageAssessmentDatset(
        str(dataset_root), val_ids, crop_size=crop_size, max_iters=None, type="test"
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    evaluator_loc = Evaluator(num_class=2)
    evaluator_clf = Evaluator(num_class=4)
    evaluator_total = Evaluator(num_class=4)
    started = time.perf_counter()
    with torch.inference_mode():
        for pre, post, labels_loc, labels_clf, _sample_ids in loader:
            pre = pre.to(device)
            post = post.to(device)
            labels_loc_np = labels_loc.numpy()
            labels_clf_device = labels_clf.to(device).long()
            inputs = torch.cat([pre, post], dim=1)
            logits = model(inputs)
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError("Non-finite logits during validation")
            prediction = torch.argmax(logits, dim=1).cpu().numpy()
            labels_clf_np = labels_clf_device.cpu().numpy()
            prediction_loc = prediction.copy()
            prediction_loc[prediction_loc > 0] = 1
            evaluator_loc.add_batch(labels_loc_np, prediction_loc)
            evaluator_clf.add_batch(
                labels_clf_np[labels_loc_np > 0], prediction[labels_loc_np > 0]
            )
            evaluator_total.add_batch(labels_clf_np, prediction)

    report = metrics_from_evaluators(evaluator_loc, evaluator_clf, evaluator_total)
    report.update(
        {
            "sample_count": len(val_ids),
            "split": "official standard_ML val only",
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    model.train()
    return report


def checkpoint_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    state: dict[str, Any],
    config: dict[str, Any],
    split_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "training_state": state,
        "configuration": config,
        "split_sha256": split_hashes,
        "git_commit": git_commit(),
        "rng_state": capture_rng(),
    }


def save_full_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    state: dict[str, Any],
    config: dict[str, Any],
    split_hashes: dict[str, str],
) -> None:
    atomic_torch_save(
        checkpoint_state(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=state,
            config=config,
            split_hashes=split_hashes,
        ),
        path,
    )


def safe_grad_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    bad: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        if not bool(finite.all().item()):
            bad[name] = {
                "numel": gradient.numel(),
                "nan_count": int(torch.isnan(gradient).sum().item()),
                "positive_inf_count": int(torch.isposinf(gradient).sum().item()),
                "negative_inf_count": int(torch.isneginf(gradient).sum().item()),
            }
    return {"all_finite": not bad, "non_finite_parameters": bad}


def normalize_accumulated_gradients(model: torch.nn.Module, divisor: int) -> None:
    """Average accumulated per-microbatch mean gradients, including a short tail."""
    if divisor < 1:
        raise ValueError("Gradient normalization divisor must be positive")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(divisor)


def save_raw_state_dict(model: torch.nn.Module, path: Path) -> None:
    atomic_torch_save(model.state_dict(), path)


def load_resume(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: dict[str, Any],
    split_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch_load_full(path)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported training checkpoint version")
    if payload.get("configuration") != config:
        raise ValueError("Resume configuration differs from checkpoint")
    if payload.get("split_sha256") != split_hashes:
        raise ValueError("Official train/val split hash differs from checkpoint")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler.load_state_dict(payload["grad_scaler_state_dict"])
    restore_rng(payload["rng_state"])
    return payload["training_state"], payload["rng_state"]


def validate_dataset_triplets(dataset_root: Path, sample_ids: list[str]) -> None:
    missing: list[str] = []
    for sample_id in sample_ids:
        paths = (
            dataset_root / "pre-event" / f"{sample_id}_pre_disaster.tif",
            dataset_root / "post-event" / f"{sample_id}_post_disaster.tif",
            dataset_root / "target" / f"{sample_id}_building_damage.tif",
        )
        missing.extend(str(path) for path in paths if not path.is_file())
        if len(missing) >= 20:
            break
    if missing:
        raise FileNotFoundError("Missing official train/val TIFF files:\n" + "\n".join(missing))


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    validate_args(args, device)
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; choose a new run directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = read_splits()
    validate_dataset_triplets(args.dataset_root, splits["train"] + splits["val"])
    split_hashes = {
        "train": sha256_file(TRAIN_LIST),
        "val": sha256_file(VAL_LIST),
    }
    split_id_hashes = {
        "train": sha256_lines(splits["train"]),
        "val": sha256_lines(splits["val"]),
    }
    amp_enabled = bool(args.amp and device.type == "cuda")
    config = critical_config(args, amp_enabled)
    config_path = args.output_dir / "config.json"
    hashes_path = args.output_dir / "split_hashes.json"
    hashes_document = {
        "source_file_sha256": split_hashes,
        "ordered_ids_sha256": split_id_hashes,
    }
    if args.resume:
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing run config differs from requested resume config")
        if hashes_path.exists() and json.loads(hashes_path.read_text(encoding="utf-8")) != hashes_document:
            raise ValueError("Existing run split hashes differ from current official lists")
    if not config_path.exists():
        atomic_write_json(config_path, config)
    if not hashes_path.exists():
        atomic_write_json(hashes_path, hashes_document)
    class_weight_metadata = getattr(args, "class_weight_metadata", None)
    if class_weight_metadata is not None:
        class_weights_path = args.output_dir / "class_weights.json"
        if not class_weights_path.exists():
            atomic_write_json(class_weights_path, class_weight_metadata)

    set_seed(args.seed)
    model = UNet(in_channels=6, num_classes=4).to(device)
    initial_checkpoint = getattr(args, "initial_checkpoint", None)
    if initial_checkpoint and not args.resume:
        initial_path = Path(initial_checkpoint)
        payload = torch_load_full(initial_path)
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in payload.items()
        ):
            raise TypeError("Initial checkpoint must be a raw state_dict mapping")
        model.load_state_dict(payload, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    dataset = MultimodalDamageAssessmentDatset(
        str(args.dataset_root),
        splits["train"],
        crop_size=args.crop_size,
        max_iters=args.sample_exposures,
        type="train",
    )
    sampler_seed = args.seed + 2
    resume_state: dict[str, Any] | None = None
    resume_rng_state: dict[str, Any] | None = None
    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint missing: {args.resume}")
        resume_state, resume_rng_state = load_resume(
            args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            split_hashes=split_hashes,
        )
        sampler_seed = int(resume_state["sampler_seed"])

    state = resume_state or {
        "samples_seen": 0,
        "micro_steps": 0,
        "optimizer_update_attempts": 0,
        "successful_optimizer_updates": 0,
        "amp_overflow_skips": 0,
        "consecutive_amp_skips": 0,
        "best_val_miou": -1.0,
        "best_update": None,
        "last_validation_update": 0,
        "last_validation_samples_seen": 0,
        "sampler_seed": sampler_seed,
        "elapsed_training_seconds": 0.0,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "step_time_seconds": [],
        "validation_history": [],
    }
    state.setdefault("last_validation_samples_seen", 0)
    state.setdefault("validated_sample_exposures", [])
    state.setdefault("peak_allocated_bytes", 0)
    state.setdefault("peak_reserved_bytes", 0)
    if state["samples_seen"] > args.sample_exposures:
        raise ValueError("Checkpoint sample cursor exceeds the configured budget")
    if state["samples_seen"] % args.micro_batch_size:
        raise ValueError("Checkpoint cursor is not at a micro-batch boundary")
    elapsed_before_run = float(state["elapsed_training_seconds"])

    sampler = ResumableRandomSampler(
        len(dataset), sampler_seed, start_index=int(state["samples_seen"])
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
    )
    loader_iterator = iter(loader)
    # Restore after creating the iterator: with num_workers=0 this iterator does
    # not preload dataset items, and the augmentation RNG resumes at the exact
    # saved optimizer boundary.
    if resume_rng_state:
        restore_rng(resume_rng_state)

    model.train()
    if not args.resume:
        save_full_checkpoint(
            args.output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=state,
            config=config,
            split_hashes=split_hashes,
        )
    log_path = args.output_dir / getattr(args, "training_log_filename", "train.jsonl")
    validation_log_path = args.output_dir / getattr(
        args, "validation_log_filename", "validation.jsonl"
    )
    log_mode = "a" if args.resume else "w"
    optimizer.zero_grad(set_to_none=True)
    class_weights = getattr(args, "class_weights", None)
    ce_weight_tensor = (
        torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        if class_weights is not None
        else None
    )
    pending_micro_steps = 0
    started = time.perf_counter()
    active_step_context: dict[str, Any] = {}

    try:
        with log_path.open(log_mode, encoding="utf-8") as train_log:
            while state["samples_seen"] < args.sample_exposures:
                step_started = time.perf_counter()
                pre, post, _labels_loc, labels_clf, sample_ids = next(loader_iterator)
                state["samples_seen"] += len(sample_ids)
                state["micro_steps"] += 1
                active_step_context = {
                    "micro_step": state["micro_steps"],
                    "sample_ids": list(sample_ids),
                    "samples_seen": state["samples_seen"],
                    "stage": "input_validation_before_forward",
                }
                inputs = torch.cat([pre, post], dim=1).to(device)
                labels_clf = labels_clf.to(device).long()
                actual_batch_size = len(sample_ids)
                expected_input = (
                    actual_batch_size,
                    6,
                    args.crop_size,
                    args.crop_size,
                )
                if tuple(inputs.shape) != expected_input:
                    raise ValueError(
                        f"Input shape {tuple(inputs.shape)} != {expected_input}"
                    )
                label_values = sorted(
                    int(value) for value in torch.unique(labels_clf).tolist()
                )
                invalid = sorted(set(label_values) - {0, 1, 2, 3, 255})
                if invalid:
                    raise ValueError(f"Unexpected train label values: {invalid}")

                if pending_micro_steps == 0 and device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    active_step_context["stage"] = "autocast_forward_and_loss"
                    logits = model(inputs)
                    if tuple(logits.shape) != (
                        actual_batch_size,
                        4,
                        args.crop_size,
                        args.crop_size,
                    ):
                        raise ValueError(f"Unexpected output shape: {tuple(logits.shape)}")
                    ce_loss = F.cross_entropy(
                        logits, labels_clf, weight=ce_weight_tensor
                    )
                    lovasz_loss = lovasz_softmax(
                        F.softmax(logits, dim=1), labels_clf, ignore=255
                    )
                    total_loss = ce_loss + 0.75 * lovasz_loss

                ce_value = float(ce_loss.detach().item())
                lovasz_value = float(lovasz_loss.detach().item())
                loss_value = float(total_loss.detach().item())
                active_step_context["losses"] = {
                    "ce": ce_value if math.isfinite(ce_value) else None,
                    "lovasz": lovasz_value if math.isfinite(lovasz_value) else None,
                    "total": loss_value if math.isfinite(loss_value) else None,
                    "ce_finite": math.isfinite(ce_value),
                    "lovasz_finite": math.isfinite(lovasz_value),
                    "total_finite": math.isfinite(loss_value),
                }
                if not all(math.isfinite(value) for value in (ce_value, lovasz_value, loss_value)):
                    raise FloatingPointError(
                        "Non-finite CE/Lovasz/total loss; training stopped without "
                        "optimizer update"
                    )
                active_step_context["stage"] = "backward"
                scaler.scale(total_loss).backward()
                pending_micro_steps += 1
                record: dict[str, Any] = {
                    "micro_step": state["micro_steps"],
                    "sample_ids": list(sample_ids),
                    "events": [event_for_sample(sample_id) for sample_id in sample_ids],
                    "samples_seen": state["samples_seen"],
                    "label_values": label_values,
                    "ce_loss": ce_value,
                    "lovasz_loss": lovasz_value,
                    "weighted_lovasz_loss": 0.75 * lovasz_value,
                    "total_loss": loss_value,
                    "amp": amp_enabled,
                    "grad_scaler_scale": float(scaler.get_scale()),
                    "input_shape": list(inputs.shape),
                    "output_shape": list(logits.shape),
                    "micro_step_seconds_including_data_and_backward": None,
                }

                is_budget_end = state["samples_seen"] >= args.sample_exposures
                if (
                    pending_micro_steps == args.gradient_accumulation_steps
                    or is_budget_end
                ):
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                    normalize_accumulated_gradients(model, pending_micro_steps)
                    scale_before = float(scaler.get_scale()) if amp_enabled else 1.0
                    active_step_context["grad_scaler_scale_before_step"] = scale_before
                    active_step_context["stage"] = "optimizer_step_and_scaler_update"
                    if not amp_enabled:
                        gradient_report = safe_grad_diagnostics(model)
                        if not gradient_report["all_finite"]:
                            record["gradient_diagnostics"] = gradient_report
                            raise FloatingPointError(
                                "Non-finite FP32 gradient before optimizer update"
                            )
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                        scale_after = float(scaler.get_scale())
                        amp_skipped = scale_after < scale_before
                    else:
                        optimizer.step()
                        scale_after = 1.0
                        amp_skipped = False

                    state["optimizer_update_attempts"] += 1
                    active_step_context["grad_scaler_scale_after_update"] = scale_after
                    if device.type == "cuda":
                        group_peak_allocated = int(
                            torch.cuda.max_memory_allocated(device)
                        )
                        group_peak_reserved = int(
                            torch.cuda.max_memory_reserved(device)
                        )
                        state["peak_allocated_bytes"] = max(
                            state["peak_allocated_bytes"], group_peak_allocated
                        )
                        state["peak_reserved_bytes"] = max(
                            state["peak_reserved_bytes"], group_peak_reserved
                        )
                        record_memory = {
                            "step_peak_allocated_bytes": group_peak_allocated,
                            "step_peak_reserved_bytes": group_peak_reserved,
                        }
                    else:
                        record_memory = None
                    if amp_skipped:
                        state["amp_overflow_skips"] += 1
                        state["consecutive_amp_skips"] += 1
                        gradient_report = safe_grad_diagnostics(model)
                        active_step_context["gradient_diagnostics"] = gradient_report
                        if gradient_report["all_finite"]:
                            raise RuntimeError(
                                "GradScaler reduced its scale but post-unscale gradients "
                                "were all finite; stopping because the skipped update "
                                "could not be attributed to gradient overflow"
                            )
                    else:
                        state["successful_optimizer_updates"] += 1
                        state["consecutive_amp_skips"] = 0
                        gradient_report = {"all_finite": True}
                    optimizer.zero_grad(set_to_none=True)
                    pending_micro_steps = 0
                    record.update(
                        {
                            "optimizer_update_attempt": state[
                                "optimizer_update_attempts"
                            ],
                            "successful_optimizer_updates": state[
                                "successful_optimizer_updates"
                            ],
                            "amp_overflow_skipped_update": amp_skipped,
                            "grad_scaler_scale_before_step": scale_before,
                            "grad_scaler_scale_after_update": scale_after,
                            "cuda_memory_for_accumulation_group": record_memory,
                            "gradient_diagnostics_on_overflow": gradient_report
                            if amp_skipped
                            else None,
                        }
                    )
                    record["micro_step_seconds_including_data_and_backward"] = (
                        time.perf_counter() - step_started
                    )

                    if state["consecutive_amp_skips"] >= args.max_consecutive_amp_skips:
                        state["elapsed_training_seconds"] = (
                            elapsed_before_run + time.perf_counter() - started
                        )
                        record["stop_reason"] = "consecutive_amp_overflow_limit"
                        record["micro_step_seconds_including_data_and_backward"] = (
                            time.perf_counter() - step_started
                        )
                        train_log.write(json.dumps(record, ensure_ascii=False) + "\n")
                        train_log.flush()
                        save_full_checkpoint(
                            args.output_dir / "latest.pt",
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            state=state,
                            config=config,
                            split_hashes=split_hashes,
                        )
                        raise RuntimeError(
                            f"Stopped after {state['consecutive_amp_skips']} consecutive "
                            "GradScaler-skipped updates; latest.pt is resumable"
                        )

                    validation_milestones = getattr(
                        args, "validation_sample_exposures", []
                    )
                    due_milestones = [
                        milestone
                        for milestone in validation_milestones
                        if milestone <= state["samples_seen"]
                        and milestone not in state["validated_sample_exposures"]
                    ]
                    regular_validation_due = (
                        state["successful_optimizer_updates"] > 0
                        and state["last_validation_update"]
                        < state["successful_optimizer_updates"]
                        and state["successful_optimizer_updates"]
                        % args.validation_every_updates
                        == 0
                    )
                    if due_milestones or regular_validation_due:
                        val_metrics = evaluate_validation(
                            model=model,
                            dataset_root=args.dataset_root,
                            val_ids=splits["val"],
                            crop_size=1024,
                            device=device,
                        )
                        val_metrics["optimizer_update"] = state[
                            "successful_optimizer_updates"
                        ]
                        val_metrics["samples_seen"] = state["samples_seen"]
                        val_metrics["requested_sample_exposure_milestones"] = due_milestones
                        state["validated_sample_exposures"].extend(due_milestones)
                        state["last_validation_update"] = state[
                            "successful_optimizer_updates"
                        ]
                        state["last_validation_samples_seen"] = state["samples_seen"]
                        state["validation_history"].append(val_metrics)
                        with validation_log_path.open("a", encoding="utf-8") as val_log:
                            val_log.write(json.dumps(val_metrics, ensure_ascii=False) + "\n")
                        miou = val_metrics["miou"]
                        if miou is not None and miou > state["best_val_miou"]:
                            state["best_val_miou"] = miou
                            state["best_update"] = state[
                                "successful_optimizer_updates"
                            ]
                            save_raw_state_dict(
                                model, args.output_dir / "best_model.pth"
                            )
                            save_full_checkpoint(
                                args.output_dir / "best_training_state.pt",
                                model=model,
                                optimizer=optimizer,
                                scaler=scaler,
                                state=state,
                                config=config,
                                split_hashes=split_hashes,
                            )
                        print(
                            "validation "
                            f"update={state['successful_optimizer_updates']} "
                            f"mIoU={val_metrics['miou']} "
                            f"best={state['best_val_miou']:.6f}",
                            flush=True,
                        )

                    if (
                        state["optimizer_update_attempts"]
                        % args.checkpoint_every_updates
                        == 0
                    ):
                        state["elapsed_training_seconds"] = (
                            elapsed_before_run + time.perf_counter() - started
                        )
                        save_full_checkpoint(
                            args.output_dir / "latest.pt",
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            state=state,
                            config=config,
                            split_hashes=split_hashes,
                        )

                if record["micro_step_seconds_including_data_and_backward"] is None:
                    record["micro_step_seconds_including_data_and_backward"] = (
                        time.perf_counter() - step_started
                    )
                state["step_time_seconds"].append(
                    record["micro_step_seconds_including_data_and_backward"]
                )
                if len(state["step_time_seconds"]) > 10000:
                    state["step_time_seconds"] = state["step_time_seconds"][-10000:]
                train_log.write(json.dumps(record, ensure_ascii=False) + "\n")
                train_log.flush()
                if state["micro_steps"] % 100 == 0:
                    print(
                        f"micro_step={state['micro_steps']} "
                        f"samples={state['samples_seen']}/{args.sample_exposures} "
                        f"updates={state['successful_optimizer_updates']}/"
                        f"{state['optimizer_update_attempts']} attempts "
                        f"loss={loss_value:.6f} scaler={scaler.get_scale():.1f} "
                        f"elapsed={elapsed_before_run + time.perf_counter() - started:.1f}s",
                        flush=True,
                    )

                if state["samples_seen"] >= args.sample_exposures:
                    break

        # The sample-exposure budget is primary. Overflow skips may leave the
        # successful update count below 50000; perform a final validation only.
        if (
            not state["validation_history"]
            or state["last_validation_samples_seen"] != state["samples_seen"]
        ):
            final_val = evaluate_validation(
                model=model,
                dataset_root=args.dataset_root,
                val_ids=splits["val"],
                crop_size=1024,
                device=device,
            )
            final_val["optimizer_update"] = state["successful_optimizer_updates"]
            final_val["final_validation"] = True
            state["last_validation_update"] = state["successful_optimizer_updates"]
            state["last_validation_samples_seen"] = state["samples_seen"]
            state["validation_history"].append(final_val)
            with validation_log_path.open("a", encoding="utf-8") as val_log:
                val_log.write(json.dumps(final_val, ensure_ascii=False) + "\n")
            miou = final_val["miou"]
            if miou is not None and miou > state["best_val_miou"]:
                state["best_val_miou"] = miou
                state["best_update"] = state["successful_optimizer_updates"]
                save_raw_state_dict(model, args.output_dir / "best_model.pth")
                save_full_checkpoint(
                    args.output_dir / "best_training_state.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    state=state,
                    config=config,
                    split_hashes=split_hashes,
                )

        state["elapsed_training_seconds"] = (
            elapsed_before_run + time.perf_counter() - started
        )
        save_full_checkpoint(
            args.output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=state,
            config=config,
            split_hashes=split_hashes,
        )
        if getattr(args, "save_final_checkpoint", False):
            save_full_checkpoint(
                args.output_dir / "final.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                state=state,
                config=config,
                split_hashes=split_hashes,
            )
        times = state["step_time_seconds"]
        report = {
            "status": "completed_sample_exposure_budget",
            "paper_result": False,
            "experiment_mode": config["experiment_mode"],
            "paper_reproduction_status": config["paper_reproduction_status"],
            "experiment_role": config["experiment_mode"],
            "configuration": config,
            "data_access": {
                "train_list": str(TRAIN_LIST.resolve()),
                "val_list": str(VAL_LIST.resolve()),
                "train_count": len(splits["train"]),
                "val_count": len(splits["val"]),
                "test_manifest_or_images_accessed": False,
                "train_list_sha256": split_hashes["train"],
                "val_list_sha256": split_hashes["val"],
            },
            "training_state": state,
            "cuda_peak_memory_bytes": {
                "allocated": state["peak_allocated_bytes"],
                "reserved": state["peak_reserved_bytes"],
            },
            "timing": {
                "measured_micro_step_count_retained": len(times),
                "micro_step_median_seconds_including_data": (
                    statistics.median(times) if times else None
                ),
                "micro_step_p90_seconds_including_data": (
                    float(np.percentile(times, 90)) if times else None
                ),
                "note": "Timing history retains the last 10000 micro-steps; validation and checkpoint time are in elapsed_training_seconds, not per-step median.",
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else None
                ),
                "git_commit": git_commit(),
            },
            "best_checkpoint": str((args.output_dir / "best_model.pth").resolve()),
        }
        atomic_write_json(
            args.output_dir / getattr(args, "report_filename", "training_report.json"),
            report,
        )
        return report
    except Exception as error:
        if pending_micro_steps:
            optimizer.zero_grad(set_to_none=True)
        memory = None
        if torch.cuda.is_available():
            memory = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            torch.cuda.empty_cache()
        failure = {
            "status": "stopped_with_error",
            "error_type": type(error).__name__,
            "message": str(error),
            "samples_seen": state["samples_seen"],
            "micro_steps": state["micro_steps"],
            "optimizer_update_attempts": state["optimizer_update_attempts"],
            "successful_optimizer_updates": state["successful_optimizer_updates"],
            "amp_overflow_skips": state["amp_overflow_skips"],
            "consecutive_amp_skips": state["consecutive_amp_skips"],
            "pending_micro_steps_discarded": pending_micro_steps,
            "failing_step_context": active_step_context,
            "elapsed_this_invocation_seconds": time.perf_counter() - started,
            "latest_checkpoint": str((args.output_dir / "latest.pt").resolve()),
            "latest_checkpoint_exists": (args.output_dir / "latest.pt").is_file(),
            "cuda_memory": memory,
            "cuda_peak_memory_bytes": {
                "allocated": state["peak_allocated_bytes"],
                "reserved": state["peak_reserved_bytes"],
            },
            "traceback": traceback.format_exc().splitlines(),
        }
        atomic_write_json(next_failure_report_path(args.output_dir), failure)
        raise


def main() -> int:
    args = parse_args()
    existing_failure_reports = (
        {path.name for path in args.output_dir.glob("failure_report*.json")}
        if args.output_dir.exists()
        else set()
    )
    output_safe_at_start = (
        args.resume is not None
        or not args.output_dir.exists()
        or not any(args.output_dir.iterdir())
    )
    try:
        report = run(args)
    except Exception as error:
        new_failure_reports = (
            {path.name for path in args.output_dir.glob("failure_report*.json")}
            - existing_failure_reports
            if args.output_dir.exists()
            else set()
        )
        if output_safe_at_start and not new_failure_reports:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                next_failure_report_path(args.output_dir),
                {
                    "status": "stopped_with_error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc().splitlines(),
                    "latest_checkpoint_exists": (
                        args.output_dir / "latest.pt"
                    ).is_file(),
                },
            )
        raise
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Training report: {args.output_dir / 'training_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
