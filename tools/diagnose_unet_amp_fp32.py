#!/usr/bin/env python3
"""Compare AMP and FP32 finite behavior on the same standard_ML train batches.

Each branch starts from the same seeded UNet initialization and replays the
same official training IDs and random crops. It stops at the first non-finite
loss/gradient and never applies an optimizer update when gradients are bad.
This is a diagnostic replay, not a paper run; no validation/test list is read.
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


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = list(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the same official standard_ML training batches with AMP and "
            "FP32, stopping at the first non-finite value."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "standard_ml_amp_fp32_diagnostic",
    )
    parser.add_argument("--max-updates", type=int, default=250)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def event_for_sample(sample_id: str) -> str:
    if "_" not in sample_id:
        raise ValueError(f"Sample ID has no numeric suffix: {sample_id}")
    return sample_id.rsplit("_", 1)[0]


def read_train_ids() -> list[str]:
    if not TRAIN_LIST.is_file():
        raise FileNotFoundError(f"Official training list is missing: {TRAIN_LIST}")
    ids = [line.strip() for line in TRAIN_LIST.read_text(encoding="utf-8").splitlines()]
    ids = [sample_id for sample_id in ids if sample_id]
    if len(ids) != EXPECTED_TRAIN_COUNT or len(set(ids)) != EXPECTED_TRAIN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_COUNT} unique train IDs; got {len(ids)}"
        )
    events = {event_for_sample(sample_id) for sample_id in ids}
    if events != EXPECTED_EVENTS:
        raise ValueError(
            f"Training events differ: missing={sorted(EXPECTED_EVENTS-events)}, "
            f"unexpected={sorted(events-EXPECTED_EVENTS)}"
        )
    return ids


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_tensors(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_range(tensor: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(tensor)
    if not bool(finite.all().item()):
        return {
            "dtype": str(tensor.dtype),
            "finite": False,
            "nan_count": int(torch.isnan(tensor).sum().item()),
            "positive_inf_count": int(torch.isposinf(tensor).sum().item()),
            "negative_inf_count": int(torch.isneginf(tensor).sum().item()),
        }
    values = tensor.detach().float()
    return {
        "dtype": str(tensor.dtype),
        "finite": True,
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
    }


def json_scalar(value: float) -> tuple[float | None, str | None]:
    value = float(value)
    if math.isnan(value):
        return None, "nan"
    if value == math.inf:
        return None, "+inf"
    if value == -math.inf:
        return None, "-inf"
    return value, None


def gradient_diagnostics(model: torch.nn.Module) -> tuple[bool, dict[str, Any]]:
    problems: dict[str, Any] = {}
    grad_parameter_count = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        grad_parameter_count += 1
        finite_mask = torch.isfinite(gradient)
        if not bool(finite_mask.all().item()):
            problems[name] = {
                "numel": gradient.numel(),
                "finite_count": int(finite_mask.sum().item()),
                "nan_count": int(torch.isnan(gradient).sum().item()),
                "positive_inf_count": int(torch.isposinf(gradient).sum().item()),
                "negative_inf_count": int(torch.isneginf(gradient).sum().item()),
                "finite_abs_max": (
                    float(gradient[finite_mask].abs().max().item())
                    if bool(finite_mask.any().item())
                    else None
                ),
            }
    if problems:
        return False, {
            "all_gradients_finite": False,
            "parameters_with_gradients": grad_parameter_count,
            "non_finite_parameters": problems,
        }
    return True, {
        "all_gradients_finite": True,
        "parameters_with_gradients": grad_parameter_count,
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    amp_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        logits = model(inputs)
        ce_loss = F.cross_entropy(logits, labels)
        lovasz_loss = L.lovasz_softmax(
            F.softmax(logits, dim=1), labels, ignore=255
        )
        total_loss = ce_loss + 0.75 * lovasz_loss
    return logits, ce_loss, lovasz_loss, total_loss


def run_branch(
    *,
    mode: str,
    args: argparse.Namespace,
    train_ids: list[str],
    device: torch.device,
) -> dict[str, Any]:
    amp_enabled = mode == "amp"
    seed_everything(args.seed)
    model = UNet(in_channels=6, num_classes=4).to(device)
    initial_parameter_hash = hashlib.sha256(
        next(model.parameters()).detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    loader_generator = torch.Generator().manual_seed(args.seed + 1)
    order_generator = torch.Generator().manual_seed(args.seed + 2)
    sample_order = torch.randperm(len(train_ids), generator=order_generator).tolist()
    indices = sample_order[: args.max_updates * args.micro_batch_size]
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
        sampler=FixedOrderSampler(indices),
        num_workers=0,
        drop_last=True,
        generator=loader_generator,
    )

    records: list[dict[str, Any]] = []
    status = "completed_limit"
    failure: dict[str, Any] | None = None
    started = time.perf_counter()
    for update, batch in enumerate(loader, start=1):
        pre, post, _labels_loc, labels, sample_ids = batch
        inputs = torch.cat([pre, post], dim=1).to(device)
        labels = labels.to(device).long()
        input_digest = sha256_tensors(inputs, labels)
        record: dict[str, Any] = {
            "update": update,
            "sample_ids": list(sample_ids),
            "events": [event_for_sample(sample_id) for sample_id in sample_ids],
            "input_shape": list(inputs.shape),
            "input_label_sha256": input_digest,
            "stage": "forward",
            "grad_scaler_enabled": amp_enabled,
            "grad_scaler_scale_before_backward": (
                float(scaler.get_scale()) if amp_enabled else 1.0
            ),
        }
        try:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            logits, ce_loss, lovasz_loss, total_loss = loss_forward(
                model, inputs, labels, amp_enabled
            )
            record["logits"] = tensor_range(logits)
            if not record["logits"]["finite"]:
                raise FloatingPointError("Non-finite model logits")
            record["output_shape"] = list(logits.shape)
            raw_losses = {
                "ce_loss": float(ce_loss.detach().item()),
                "lovasz_loss": float(lovasz_loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
            }
            for name, value in raw_losses.items():
                record[name], record[f"{name}_non_finite_kind"] = json_scalar(value)
            record["weighted_lovasz_loss"], record[
                "weighted_lovasz_loss_non_finite_kind"
            ] = json_scalar(0.75 * raw_losses["lovasz_loss"])
            record["losses_finite"] = all(
                math.isfinite(value) for value in raw_losses.values()
            )
            if not record["losses_finite"]:
                raise FloatingPointError("Non-finite CE/Lovasz/total loss")

            record["stage"] = "scaled_backward" if amp_enabled else "backward"
            if amp_enabled:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            record["stage"] = "scaler_unscale" if amp_enabled else "gradient_check"
            if amp_enabled:
                scaler.unscale_(optimizer)
                record["grad_scaler_scale_after_unscale_before_update"] = float(
                    scaler.get_scale()
                )
            gradients_finite, grad_report = gradient_diagnostics(model)
            record["gradient_diagnostics"] = grad_report
            if not gradients_finite:
                record["stage"] = (
                    "gradient_check_after_scaler_unscale_before_scaler_step"
                    if amp_enabled
                    else "gradient_check_before_optimizer_step"
                )
                if amp_enabled:
                    # The branch terminates here. Let GradScaler report its
                    # response to the recorded inf check; no optimizer update
                    # is attempted and this model is never resumed.
                    before = float(scaler.get_scale())
                    scaler.update()
                    record["grad_scaler_scale_after_diagnostic_update"] = float(
                        scaler.get_scale()
                    )
                    record["grad_scaler_scale_decreased"] = (
                        float(scaler.get_scale()) < before
                    )
                failure = {
                    "stage": record["stage"],
                    "sample_ids": list(sample_ids),
                    "losses_finite": record.get("losses_finite"),
                    "gradient_diagnostics": grad_report,
                    "optimizer_step_reached": False,
                    "scaler_update_after_diagnostic_only": amp_enabled,
                }
                status = "stopped_non_finite_gradient"
                records.append(record)
                break

            record["stage"] = "optimizer_step"
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
                record["grad_scaler_scale_after_update"] = float(scaler.get_scale())
            else:
                optimizer.step()
            record["optimizer_step_completed"] = True
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record["elapsed_seconds"] = time.perf_counter() - started
            records.append(record)
        except Exception as error:  # noqa: BLE001 - preserve exact failing phase
            if status != "stopped_non_finite_gradient":
                status = "stopped_exception"
                failure = {
                    "stage": record.get("stage"),
                    "sample_ids": list(sample_ids),
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "optimizer_step_reached": False,
                }
                record["exception_type"] = type(error).__name__
                record["exception"] = str(error)
                records.append(record)
            break

    return {
        "mode": mode,
        "status": status,
        "initial_parameter_sha256": initial_parameter_hash,
        "updates_completed": sum(
            bool(record.get("optimizer_step_completed")) for record in records
        ),
        "attempts_recorded": len(records),
        "failure": failure,
        "steps": records,
        "peak_cuda_memory": (
            {
                "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            if device.type == "cuda"
            else None
        ),
    }


def atomic_write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.max_updates < 1 or args.micro_batch_size < 1 or args.crop_size < 1:
        raise ValueError("Update count, batch size, and crop size must be positive")
    if args.max_updates * args.micro_batch_size > EXPECTED_TRAIN_COUNT:
        raise ValueError("Diagnostic is bounded to one pass over unique train IDs")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer parameters")
    if args.device == "cpu":
        raise ValueError("AMP/FP32 CUDA comparison requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to compare AMP and FP32 on this GPU")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")

    train_ids = read_train_ids()
    device = torch.device("cuda")
    branches: dict[str, Any] = {}
    common_input_hashes: dict[str, list[str]] = {}
    for mode in ("amp", "fp32"):
        torch.cuda.reset_peak_memory_stats(device)
        branches[mode] = run_branch(
            mode=mode,
            args=args,
            train_ids=train_ids,
            device=device,
        )
        if (
            branches[mode]["status"] == "stopped_exception"
            and "out of memory"
            in str((branches[mode].get("failure") or {}).get("message", "")).lower()
        ):
            if mode == "amp":
                branches["fp32"] = {
                    "mode": "fp32",
                    "status": "not_started_after_amp_oom",
                    "failure": None,
                    "steps": [],
                }
            break
        common_input_hashes[mode] = [
            step["input_label_sha256"] for step in branches[mode]["steps"]
        ]

    matching_count = 0
    for amp_hash, fp32_hash in zip(
        common_input_hashes.get("amp", []), common_input_hashes.get("fp32", [])
    ):
        if amp_hash != fp32_hash:
            break
        matching_count += 1

    report = {
        "status": "diagnostic_complete",
        "paper_result": False,
        "experiment_role": "bounded_amp_fp32_finite_behavior_comparison",
        "data_access": {
            "training_list": str(TRAIN_LIST.resolve()),
            "training_count": len(train_ids),
            "training_event_count": len(
                {event_for_sample(sample_id) for sample_id in train_ids}
            ),
            "validation_list_accessed": False,
            "test_list_accessed": False,
        },
        "configuration": {
            "dataset_root": str(args.dataset_root.resolve()),
            "max_updates_per_branch": args.max_updates,
            "micro_batch_size": args.micro_batch_size,
            "crop_size": args.crop_size,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss": "cross_entropy + 0.75 * official lovasz_softmax",
            "seed": args.seed,
            "scheduler": None,
            "gradient_accumulation_steps": 1,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "paired_batches_with_identical_augmented_tensor_hashes": matching_count,
        },
        "interpretation": (
            "Each branch starts from the same seeded initialization and independently "
            "replays the same train IDs/crops. The branches diverge in parameters as "
            "their numerical updates differ. Non-finite gradients are recorded and "
            "never passed to optimizer.step."
        ),
        "branches": branches,
    }
    report_path = args.output_dir / "amp_fp32_diagnostic_report.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
