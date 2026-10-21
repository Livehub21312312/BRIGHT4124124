#!/usr/bin/env python3
"""Run a guarded, short UNet training trial on BRIGHT standard_ML train IDs.

This is a diagnostic tool, not the official full trainer. It reuses the
repository's unmodified Dataset, UNet, augmentation, and Lovasz loss. It has
no validation/test-list argument and never opens either list. The default run
performs 50 immediate optimizer updates with CUDA AMP and micro-batch size 1.
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
from util_func import lovasz_loss as L  # noqa: E402


TRAIN_LIST = (
    BASELINE_ROOT / "dataset" / "splitname" / "standard_ML" / "train_set.txt"
)
EXPECTED_TRAIN_COUNT = 2918
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
CHECKPOINT_FORMAT_VERSION = 1
ACTIVE_FAILURE_CONTEXT: dict[str, Any] = {}


class FixedOrderSampler(Sampler[int]):
    """Yield a precomputed order so a checkpoint records the remaining IDs."""

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = list(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a guarded 50-update diagnostic trial using only the official "
            "14-event standard_ML training IDs."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="BRIGHT root containing pre-event, post-event, and target.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "standard_ml_short_train",
    )
    parser.add_argument(
        "--optimizer-updates",
        type=int,
        default=50,
        help="Final update target; when resuming, completed updates count toward it.",
    )
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA AMP (default); use --no-amp for FP32.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume a checkpoint made by this script; update target is total updates.",
    )
    return parser.parse_args()


def event_for_sample(sample_id: str) -> str:
    if "_" not in sample_id:
        raise ValueError(f"Sample ID has no numeric suffix: {sample_id}")
    return sample_id.rsplit("_", 1)[0]


def triplet_paths(dataset_root: Path, sample_id: str) -> dict[str, Path]:
    return {
        "pre_event": dataset_root
        / "pre-event"
        / f"{sample_id}_pre_disaster.tif",
        "post_event": dataset_root
        / "post-event"
        / f"{sample_id}_post_disaster.tif",
        "target": dataset_root
        / "target"
        / f"{sample_id}_building_damage.tif",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_parameter(parameter: torch.Tensor) -> str:
    array = parameter.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def read_and_validate_train_ids() -> list[str]:
    if not TRAIN_LIST.is_file():
        raise FileNotFoundError(f"Official training list is missing: {TRAIN_LIST}")
    ids = [line.strip() for line in TRAIN_LIST.read_text(encoding="utf-8").splitlines()]
    ids = [sample_id for sample_id in ids if sample_id]
    duplicates = sorted(
        sample_id for sample_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate training IDs: {duplicates[:10]}")
    if len(ids) != EXPECTED_TRAIN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_COUNT} standard_ML training IDs, got {len(ids)}"
        )
    events = {event_for_sample(sample_id) for sample_id in ids}
    if events != EXPECTED_EVENTS:
        raise ValueError(
            f"Unexpected training events: missing={sorted(EXPECTED_EVENTS-events)}, "
            f"unexpected={sorted(events-EXPECTED_EVENTS)}"
        )
    return ids


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
    positive = (
        "optimizer_updates",
        "micro_batch_size",
        "crop_size",
        "checkpoint_every",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmup_updates < 0:
        raise ValueError("--warmup-updates must be nonnegative")
    if args.num_workers != 0:
        raise ValueError(
            "This resumable short trial requires --num-workers 0 so augmentation "
            "RNG state is not hidden in worker processes"
        )
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative")
    if args.amp and device.type != "cuda":
        raise ValueError("AMP is enabled by default; use --no-amp on CPU")


def finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"Non-finite {name}: {result}")
    return result


def json_float_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def non_finite_kind(value: float) -> str | None:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "+inf"
    if value == -math.inf:
        return "-inf"
    return None


def torch_load_full(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def make_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=enabled)


def critical_config(args: argparse.Namespace, use_amp: bool) -> dict[str, Any]:
    return {
        "micro_batch_size": args.micro_batch_size,
        "crop_size": args.crop_size,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss": "cross_entropy + 0.75 * lovasz_softmax",
        "scheduler": None,
        "gradient_accumulation_steps": 1,
        "amp": use_amp,
        "seed": args.seed,
        "num_workers": args.num_workers,
    }


def recursive_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            recursive_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            recursive_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def capture_rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": loader_generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    loader_generator.set_state(state["loader_generator"])


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    args: argparse.Namespace,
    use_amp: bool,
    completed_updates: int,
    samples_seen: int,
    sample_order: list[int],
    next_sample_position: int,
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "purpose": "BRIGHT standard_ML diagnostic short training; not paper result",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "completed_optimizer_updates": completed_updates,
        "samples_seen": samples_seen,
        "sample_order": sample_order,
        "next_sample_position": next_sample_position,
        "rng_state": capture_rng_state(loader_generator),
        "configuration": critical_config(args, use_amp),
        "train_list": {
            "path": str(TRAIN_LIST.resolve()),
            "sha256": sha256_file(TRAIN_LIST),
            "count": EXPECTED_TRAIN_COUNT,
        },
    }


def save_checkpoint(path: Path, **kwargs: Any) -> float:
    started = time.perf_counter()
    atomic_torch_save(checkpoint_payload(**kwargs), path)
    return time.perf_counter() - started


def validate_resume_payload(
    payload: dict[str, Any], args: argparse.Namespace, use_amp: bool
) -> None:
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported short-training checkpoint format")
    expected = critical_config(args, use_amp)
    if payload.get("configuration") != expected:
        raise ValueError(
            "Resume configuration mismatch. Saved="
            f"{payload.get('configuration')}, requested={expected}"
        )
    train_info = payload.get("train_list", {})
    if train_info.get("sha256") != sha256_file(TRAIN_LIST):
        raise ValueError("Official training-list SHA-256 differs from checkpoint")
    completed = int(payload.get("completed_optimizer_updates", -1))
    if completed < 0 or completed > args.optimizer_updates:
        raise ValueError(
            f"Checkpoint has {completed} updates but target is {args.optimizer_updates}"
        )


def verify_checkpoint_restore(
    path: Path, args: argparse.Namespace, use_amp: bool
) -> dict[str, Any]:
    """Restore into fresh CPU objects and compare all three requested states."""

    started = time.perf_counter()
    saved = torch_load_full(path, "cpu")
    verify_model = UNet(in_channels=6, num_classes=4)
    verify_optimizer = torch.optim.AdamW(
        verify_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    verify_scaler = make_scaler(use_amp)
    verify_model.load_state_dict(saved["model_state_dict"], strict=True)
    verify_optimizer.load_state_dict(saved["optimizer_state_dict"])
    verify_scaler.load_state_dict(saved["amp_scaler_state_dict"])

    checks = {
        "model_strict_load_and_exact_state": recursive_equal(
            saved["model_state_dict"], verify_model.state_dict()
        ),
        "optimizer_state_exact": recursive_equal(
            saved["optimizer_state_dict"], verify_optimizer.state_dict()
        ),
        "amp_scaler_state_exact": recursive_equal(
            saved["amp_scaler_state_dict"], verify_scaler.state_dict()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Checkpoint restore verification failed: {checks}")
    return {
        "status": "passed",
        **checks,
        "completed_optimizer_updates": saved["completed_optimizer_updates"],
        "elapsed_seconds": time.perf_counter() - started,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_summary(records: list[dict[str, Any]], warmup_updates: int) -> dict[str, Any]:
    stable = [record for record in records if record["optimizer_update"] > warmup_updates]
    used_fallback = False
    if not stable:
        stable = records
        used_fallback = True

    def summarize(key: str) -> dict[str, float]:
        values = [float(record[key]) for record in stable]
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p90": percentile(values, 0.90),
            "minimum": min(values),
            "maximum": max(values),
        }

    cycle = summarize("cycle_seconds")
    median_cycle = cycle["median"]
    return {
        "warmup_updates_excluded": warmup_updates,
        "stable_update_count": len(stable),
        "fallback_to_all_updates": used_fallback,
        "data_seconds": summarize("data_seconds"),
        "compute_seconds": summarize("compute_seconds"),
        "cycle_seconds_data_plus_compute": cycle,
        "raw_extrapolations_from_stable_cycle_median": {
            "50_micro_batch_1_updates_seconds": 50 * median_cycle,
            "1000_micro_batch_1_updates_seconds": 1000 * median_cycle,
            "warning": (
                "Excludes validation/checkpoint/startup overhead and cannot estimate "
                "true batch-16 update time."
            ),
        },
    }


def ensure_new_output_is_safe(args: argparse.Namespace) -> None:
    if args.resume:
        return
    protected = (
        args.output_dir / "short_training_report.json",
        args.output_dir / "training_log.jsonl",
    )
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to mix a new run with existing outputs: " + ", ".join(existing)
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    ACTIVE_FAILURE_CONTEXT.clear()
    invocation_started = time.perf_counter()
    device = resolve_device(args.device)
    validate_args(args, device)
    ensure_new_output_is_safe(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_ids = read_and_validate_train_ids()
    use_amp = bool(args.amp and device.type == "cuda")

    model = UNet(in_channels=6, num_classes=4).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = make_scaler(use_amp)
    loader_generator = torch.Generator().manual_seed(args.seed + 1)
    order_generator = torch.Generator().manual_seed(args.seed + 2)
    sample_order = torch.randperm(len(train_ids), generator=order_generator).tolist()
    completed_updates = 0
    samples_seen = 0
    next_sample_position = 0

    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
        # Keep serialized RNG byte tensors on CPU. load_state_dict copies model and
        # optimizer tensors to their CUDA parameter devices as needed.
        payload = torch_load_full(args.resume, "cpu")
        validate_resume_payload(payload, args, use_amp)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scaler.load_state_dict(payload["amp_scaler_state_dict"])
        completed_updates = int(payload["completed_optimizer_updates"])
        samples_seen = int(payload["samples_seen"])
        sample_order = list(payload["sample_order"])
        next_sample_position = int(payload["next_sample_position"])
        restore_rng_state(payload["rng_state"], loader_generator)

    remaining_updates = args.optimizer_updates - completed_updates
    required_samples = remaining_updates * args.micro_batch_size
    last_sample_position = next_sample_position + required_samples
    if last_sample_position > len(sample_order):
        raise ValueError(
            "This short-run tool does not cycle the dataset: requested remaining "
            f"samples end at {last_sample_position}, but train count is {len(sample_order)}"
        )

    run_indices = sample_order[next_sample_position:last_sample_position]
    missing = [
        str(path)
        for index in run_indices
        for path in triplet_paths(args.dataset_root, train_ids[index]).values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Short-run sample order references missing TIFF files:\n"
            + "\n".join(missing[:20])
        )

    dataset = MultimodalDamageAssessmentDatset(
        str(args.dataset_root),
        train_ids,
        crop_size=args.crop_size,
        max_iters=None,
        type="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=FixedOrderSampler(run_indices),
        num_workers=0,
        drop_last=True,
        generator=loader_generator,
    )
    loader_iterator = iter(loader)

    tracked_parameter_name, tracked_parameter = next(iter(model.named_parameters()))
    initial_parameter = tracked_parameter.detach().cpu().clone()
    initial_parameter_sha256 = sha256_parameter(tracked_parameter)
    log_path = args.output_dir / "training_log.jsonl"
    checkpoint_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    global_peak_allocated = 0
    global_peak_reserved = 0

    log_mode = "a" if args.resume else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_stream:
        for _ in range(remaining_updates):
            cycle_started = time.perf_counter()
            batch = next(loader_iterator)
            data_seconds = time.perf_counter() - cycle_started
            pre, post, _labels_loc, labels_clf, sample_ids = batch
            ACTIVE_FAILURE_CONTEXT.clear()
            ACTIVE_FAILURE_CONTEXT.update(
                {
                    "attempted_optimizer_update": completed_updates + 1,
                    "sample_ids": list(sample_ids),
                    "events": [event_for_sample(sample_id) for sample_id in sample_ids],
                    "stage": "input_validation_before_forward",
                    "optimizer_step_reached": False,
                    "scaler_update_reached": False,
                }
            )

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            compute_started = time.perf_counter()
            inputs = torch.cat([pre, post], dim=1).to(device)
            labels_clf = labels_clf.to(device).long()
            expected_input = (
                len(sample_ids),
                6,
                args.crop_size,
                args.crop_size,
            )
            if tuple(inputs.shape) != expected_input:
                raise ValueError(
                    f"Unexpected input shape {tuple(inputs.shape)}; expected {expected_input}"
                )
            label_values = sorted(
                int(value) for value in torch.unique(labels_clf).tolist()
            )
            invalid_labels = sorted(set(label_values) - {0, 1, 2, 3})
            if invalid_labels:
                raise ValueError(f"Invalid label values: {invalid_labels}")

            optimizer.zero_grad(set_to_none=True)
            ACTIVE_FAILURE_CONTEXT["stage"] = "autocast_forward_and_loss"
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs)
                expected_output = (
                    len(sample_ids),
                    4,
                    args.crop_size,
                    args.crop_size,
                )
                if tuple(logits.shape) != expected_output:
                    raise ValueError(
                        f"Unexpected output shape {tuple(logits.shape)}; "
                        f"expected {expected_output}"
                    )
                logits_finite = bool(torch.isfinite(logits).all().item())
                if not logits_finite:
                    ACTIVE_FAILURE_CONTEXT["logits"] = {
                        "dtype": str(logits.dtype),
                        "finite": False,
                        "nan_count": int(torch.isnan(logits).sum().item()),
                        "positive_inf_count": int(torch.isposinf(logits).sum().item()),
                        "negative_inf_count": int(torch.isneginf(logits).sum().item()),
                    }
                    raise FloatingPointError("Model output contains NaN or infinity")
                ACTIVE_FAILURE_CONTEXT["logits"] = {
                    "dtype": str(logits.dtype),
                    "finite": True,
                    "minimum": float(logits.detach().float().min().item()),
                    "maximum": float(logits.detach().float().max().item()),
                }
                ce_loss = F.cross_entropy(logits, labels_clf)
                lovasz_loss = L.lovasz_softmax(
                    F.softmax(logits, dim=1), labels_clf, ignore=255
                )
                total_loss = ce_loss + 0.75 * lovasz_loss

            raw_ce = float(ce_loss.detach().item())
            raw_lovasz = float(lovasz_loss.detach().item())
            raw_total = float(total_loss.detach().item())
            ACTIVE_FAILURE_CONTEXT["losses"] = {
                "ce": json_float_or_none(raw_ce),
                "lovasz": json_float_or_none(raw_lovasz),
                "weighted_lovasz": json_float_or_none(0.75 * raw_lovasz),
                "total": json_float_or_none(raw_total),
                "ce_finite": math.isfinite(raw_ce),
                "lovasz_finite": math.isfinite(raw_lovasz),
                "total_finite": math.isfinite(raw_total),
                "ce_non_finite_kind": non_finite_kind(raw_ce),
                "lovasz_non_finite_kind": non_finite_kind(raw_lovasz),
                "total_non_finite_kind": non_finite_kind(raw_total),
            }
            finite_float(raw_ce, "CE loss")
            finite_float(raw_lovasz, "Lovasz loss")
            finite_float(raw_total, "total loss")
            ACTIVE_FAILURE_CONTEXT["losses"]["all_finite"] = True
            scaler_scale_before = float(scaler.get_scale()) if use_amp else 1.0
            ACTIVE_FAILURE_CONTEXT["grad_scaler"] = {
                "enabled": use_amp,
                "scale_before_backward": scaler_scale_before,
                "scale_after_unscale_before_update": None,
            }
            ACTIVE_FAILURE_CONTEXT["stage"] = "scaled_backward"
            scaler.scale(total_loss).backward()
            ACTIVE_FAILURE_CONTEXT["stage"] = "scaler_unscale"
            scaler.unscale_(optimizer)
            ACTIVE_FAILURE_CONTEXT["grad_scaler"][
                "scale_after_unscale_before_update"
            ] = float(scaler.get_scale()) if use_amp else 1.0
            ACTIVE_FAILURE_CONTEXT["stage"] = (
                "gradient_check_after_scaler_unscale_before_scaler_step"
            )
            gradient_diagnostics: dict[str, Any] = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                finite_mask = torch.isfinite(parameter.grad)
                if not bool(finite_mask.all().item()):
                    gradient = parameter.grad.detach()
                    gradient_diagnostics[name] = {
                        "numel": gradient.numel(),
                        "finite_count": int(finite_mask.sum().item()),
                        "nan_count": int(torch.isnan(gradient).sum().item()),
                        "positive_inf_count": int(
                            torch.isposinf(gradient).sum().item()
                        ),
                        "negative_inf_count": int(
                            torch.isneginf(gradient).sum().item()
                        ),
                        "finite_abs_max": (
                            float(gradient[finite_mask].abs().max().item())
                            if bool(finite_mask.any().item())
                            else None
                        ),
                    }
                    ACTIVE_FAILURE_CONTEXT["gradient_diagnostics"] = (
                        gradient_diagnostics
                    )
                    raise FloatingPointError(
                        f"Non-finite gradient in parameter {name}"
                    )
            ACTIVE_FAILURE_CONTEXT["gradient_diagnostics"] = {
                "all_gradients_finite": True,
                "parameters_with_gradients": sum(
                    parameter.grad is not None for parameter in model.parameters()
                ),
            }
            ACTIVE_FAILURE_CONTEXT["stage"] = "scaler_step"
            scaler.step(optimizer)
            ACTIVE_FAILURE_CONTEXT["optimizer_step_reached"] = True
            ACTIVE_FAILURE_CONTEXT["stage"] = "scaler_update"
            scaler.update()
            ACTIVE_FAILURE_CONTEXT["scaler_update_reached"] = True
            scaler_scale_after = float(scaler.get_scale()) if use_amp else 1.0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_seconds = time.perf_counter() - compute_started
            cycle_seconds = time.perf_counter() - cycle_started

            completed_updates += 1
            batch_count = len(sample_ids)
            samples_seen += batch_count
            next_sample_position += batch_count
            if device.type == "cuda":
                step_peak_allocated = int(torch.cuda.max_memory_allocated(device))
                step_peak_reserved = int(torch.cuda.max_memory_reserved(device))
                current_allocated = int(torch.cuda.memory_allocated(device))
                current_reserved = int(torch.cuda.memory_reserved(device))
                global_peak_allocated = max(global_peak_allocated, step_peak_allocated)
                global_peak_reserved = max(global_peak_reserved, step_peak_reserved)
            else:
                step_peak_allocated = step_peak_reserved = 0
                current_allocated = current_reserved = 0

            record = {
                "optimizer_update": completed_updates,
                "sample_ids": list(sample_ids),
                "events": [event_for_sample(sample_id) for sample_id in sample_ids],
                "input_shape": list(inputs.shape),
                "output_shape": list(logits.shape),
                "label_values": label_values,
                "ce_loss": finite_float(ce_loss.detach().item(), "CE loss"),
                "lovasz_loss": finite_float(
                    lovasz_loss.detach().item(), "Lovasz loss"
                ),
                "total_loss_ce_plus_0_75_lovasz": finite_float(
                    total_loss.detach().item(), "total loss"
                ),
                "gradients_finite_after_unscale": True,
                "grad_scaler_scale_before_backward": scaler_scale_before,
                "grad_scaler_scale_after_update": scaler_scale_after,
                "data_seconds": data_seconds,
                "compute_seconds": compute_seconds,
                "cycle_seconds": cycle_seconds,
                "cuda_memory": {
                    "current_allocated_bytes": current_allocated,
                    "current_reserved_bytes": current_reserved,
                    "step_peak_allocated_bytes": step_peak_allocated,
                    "step_peak_reserved_bytes": step_peak_reserved,
                },
            }
            step_records.append(record)
            log_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_stream.flush()
            print(
                f"update={completed_updates}/{args.optimizer_updates} "
                f"loss={record['total_loss_ce_plus_0_75_lovasz']:.6f} "
                f"cycle={cycle_seconds:.3f}s"
            )

            should_checkpoint = (
                completed_updates % args.checkpoint_every == 0
                or completed_updates == args.optimizer_updates
            )
            if should_checkpoint:
                checkpoint_path = (
                    args.output_dir / f"checkpoint_step_{completed_updates:06d}.pt"
                )
                duration = save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    args=args,
                    use_amp=use_amp,
                    completed_updates=completed_updates,
                    samples_seen=samples_seen,
                    sample_order=sample_order,
                    next_sample_position=next_sample_position,
                    loader_generator=loader_generator,
                )
                checkpoint_records.append(
                    {
                        "path": str(checkpoint_path.resolve()),
                        "optimizer_update": completed_updates,
                        "save_seconds": duration,
                        "bytes": checkpoint_path.stat().st_size,
                    }
                )

    if remaining_updates == 0:
        final_checkpoint = args.resume
    else:
        final_checkpoint = args.output_dir / f"checkpoint_step_{completed_updates:06d}.pt"
    if final_checkpoint is None or not final_checkpoint.is_file():
        raise RuntimeError("Final checkpoint was not created")

    parameter_max_abs_change = float(
        (tracked_parameter.detach().cpu() - initial_parameter).abs().max().item()
    )
    parameter_changed = sha256_parameter(tracked_parameter) != initial_parameter_sha256
    if remaining_updates > 0 and (
        not parameter_changed
        or not math.isfinite(parameter_max_abs_change)
        or parameter_max_abs_change == 0
    ):
        raise RuntimeError("Tracked model parameter did not change during this invocation")

    restore_verification = verify_checkpoint_restore(
        final_checkpoint, args=args, use_amp=use_amp
    )
    counts = Counter(event_for_sample(sample_id) for sample_id in train_ids)
    report: dict[str, Any] = {
        "status": "passed",
        "experiment_role": "diagnostic_short_training_only",
        "paper_result": False,
        "scope": (
            "Official 14-event standard_ML training IDs only; no validation or test "
            "selection/evaluation."
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
                "count": len(train_ids),
                "event_counts": dict(sorted(counts.items())),
            },
            "training_samples_loaded_this_invocation": sum(
                len(record["sample_ids"]) for record in step_records
            ),
            "validation_list_accessed": False,
            "validation_tiffs_read": 0,
            "test_list_accessed": False,
            "test_tiffs_read": 0,
        },
        "configuration": {
            **critical_config(args, use_amp),
            "dataset_root": str(args.dataset_root.resolve()),
            "optimizer_update_target": args.optimizer_updates,
            "warmup_updates_for_timing": args.warmup_updates,
            "checkpoint_every": args.checkpoint_every,
            "resume_source": str(args.resume.resolve()) if args.resume else None,
            "initialization": (
                "UNet PyTorch module defaults, as in official source; deterministic "
                "seed 1337 is a reproducibility addition not reported by the paper"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "completed_optimizer_updates": completed_updates,
            "samples_seen_total": samples_seen,
            "invocation_wall_seconds_before_report_write": (
                time.perf_counter() - invocation_started
            ),
            "cuda_peak_across_recorded_steps": {
                "allocated_bytes": global_peak_allocated,
                "reserved_bytes": global_peak_reserved,
            },
            "timing": timing_summary(step_records, args.warmup_updates)
            if step_records
            else None,
        },
        "parameter_update_this_invocation": {
            "tracked_parameter": tracked_parameter_name,
            "initial_sha256": initial_parameter_sha256,
            "final_sha256": sha256_parameter(tracked_parameter),
            "max_absolute_change": parameter_max_abs_change,
            "changed": parameter_changed,
        },
        "checkpoints_saved_this_invocation": checkpoint_records,
        "final_checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_restore_verification": restore_verification,
        "step_log": str(log_path.resolve()),
    }
    atomic_write_json(args.output_dir / "short_training_report.json", report)
    return report


def failure_report(args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    oom = isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    memory: dict[str, int] | None = None
    if torch.cuda.is_available():
        memory = {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    return {
        "status": "failed",
        "experiment_role": "diagnostic_short_training_only",
        "failure_kind": "cuda_out_of_memory" if oom else type(error).__name__,
        "message": str(error),
        "stopped_without_continuing": True,
        "cuda_memory_at_failure": memory,
        "traceback": traceback.format_exc().splitlines(),
        "failing_step_context": dict(ACTIVE_FAILURE_CONTEXT),
    }


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except Exception as error:  # noqa: BLE001 - failure artifact is intentional
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = failure_report(args, error)
        atomic_write_json(args.output_dir / "failure_report.json", report)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report written to: {args.output_dir / 'short_training_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
