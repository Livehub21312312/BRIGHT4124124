#!/usr/bin/env python3
"""Run an isolated, baseline-initialized class-weighted BRIGHT UNet pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_unet_standard_ml as engine  # noqa: E402


CLASS_NAMES = ("background", "intact", "damaged", "destroyed")
DEFAULT_BASELINE = ROOT / "outputs" / "standard_ml_unet_180k_resource_adapted"
DEFAULT_OUTPUT = ROOT / "outputs" / "standard_ml_unet_class_weighted_pilot"
EXPOSURES = 20_000
VALIDATION_MILESTONES = (5_000, 10_000, 15_000, 20_000)
USER_REPORTED_BASELINE = {
    "f1loc": 0.64843,
    "f1clf": 0.0,
    "oa": None,
    "miou": 0.33855,
    "per_damage_class_f1": {
        "intact": 0.64727,
        "damaged": 0.0,
        "destroyed": 0.0,
    },
    "per_damage_class_precision": None,
    "per_damage_class_recall": None,
    "per_damage_class_iou": None,
    "best_validation_update": 4000,
    "successful_optimizer_updates": 11242,
    "sample_exposures": 180000,
    "source": "values explicitly supplied by the user in the task request",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "20k exposure class-weighted CE pilot on official standard_ML train/val; "
            "test data is never read."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-diagnostics", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def scan_train_labels(dataset_root: Path, train_ids: list[str]) -> dict[str, Any]:
    counts = np.zeros(4, dtype=np.int64)
    ignored_255 = 0
    invalid_values: Counter[int] = Counter()
    for index, sample_id in enumerate(train_ids, start=1):
        path = dataset_root / "target" / f"{sample_id}_building_damage.tif"
        if not path.is_file():
            raise FileNotFoundError(f"Missing official training label: {path}")
        label = np.asarray(imageio.imread(path))
        if label.ndim != 2:
            raise ValueError(f"Expected a 2-D target TIFF for {sample_id}; got {label.shape}")
        values, value_counts = np.unique(label, return_counts=True)
        for value, count in zip(values.tolist(), value_counts.tolist()):
            value = int(value)
            if 0 <= value <= 3:
                counts[value] += int(count)
            elif value == 255:
                ignored_255 += int(count)
            else:
                invalid_values[value] += int(count)
        if index % 250 == 0:
            print(f"scanned train labels {index}/{len(train_ids)}", flush=True)
    if invalid_values:
        raise ValueError(f"Unexpected training target values: {dict(invalid_values)}")
    total_valid = int(counts.sum())
    if total_valid == 0 or np.any(counts == 0):
        raise ValueError(f"One or more expected classes have zero train pixels: {counts.tolist()}")
    if ignored_255:
        raise ValueError(
            f"Found {ignored_255} label-255 pixels. The current baseline CE uses the "
            "official default ignore_index=-100, so it cannot safely train on these "
            "pixels without changing CE semantics. Pilot stopped before training."
        )
    frequencies = counts.astype(np.float64) / total_valid
    raw = 1.0 / np.sqrt(frequencies)

    # Find a scale so frequency-weighted mean class weight is 1 while each
    # weight stays in [0.25, 4.0]. Monotone bisection is stable for rare classes.
    lower, upper = 0.0, 1e6
    for _ in range(100):
        scale = (lower + upper) / 2.0
        candidate = np.clip(scale * raw, 0.25, 4.0)
        if float(np.dot(frequencies, candidate)) < 1.0:
            lower = scale
        else:
            upper = scale
    weights = np.clip(((lower + upper) / 2.0) * raw, 0.25, 4.0)
    return {
        "class_names_and_ids": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "training_sample_count": len(train_ids),
        "valid_class_pixel_counts": {
            name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)
        },
        "class_pixel_frequencies": {
            name: float(frequencies[i]) for i, name in enumerate(CLASS_NAMES)
        },
        "total_valid_label_pixels": total_valid,
        "raw_inverse_sqrt_weights": {
            name: float(raw[i]) for i, name in enumerate(CLASS_NAMES)
        },
        "ignored_255_pixel_count": ignored_255,
        "weight_formula": "w_i = clip(s / sqrt(freq_i), 0.25, 4.0)",
        "normalization": "s chosen so sum(freq_i * w_i) = 1",
        "weight_bounds": [0.25, 4.0],
        "class_weights": {
            name: float(weights[i]) for i, name in enumerate(CLASS_NAMES)
        },
        "class_weight_vector_id_order": [float(value) for value in weights],
        "frequency_weighted_mean_weight": float(np.dot(frequencies, weights)),
        "statistic_scope": "official standard_ML train labels only; val/test labels not scanned",
    }


def load_baseline_reference(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return dict(USER_REPORTED_BASELINE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("sample_count") != 450:
        raise ValueError("Baseline diagnostic report must cover exactly 450 official val samples")
    metrics = payload.get("official_metrics", {})
    per_class = metrics.get("per_damage_class_f1", {})
    result = dict(USER_REPORTED_BASELINE)
    result.update({
        "f1loc": metrics.get("f1loc", result["f1loc"]),
        "f1clf": metrics.get(
            "f1clf_official_harmonic_mean_damage_classes_1_to_3",
            result["f1clf"],
        ),
        "miou": metrics.get("miou", result["miou"]),
        "per_damage_class_f1": per_class or result["per_damage_class_f1"],
        "oa": metrics.get("oa", result["oa"]),
        "per_damage_class_precision": metrics.get(
            "per_damage_class_precision", result["per_damage_class_precision"]
        ),
        "per_damage_class_recall": metrics.get(
            "per_damage_class_recall", result["per_damage_class_recall"]
        ),
        "per_damage_class_iou": {
            key: value
            for key, value in metrics.get("per_class_iou", {}).items()
            if key in CLASS_NAMES[1:]
        }
        or result["per_damage_class_iou"],
        "sample_count": payload.get("sample_count"),
        "checkpoint": (payload.get("checkpoint") or {}).get("path"),
        "source": str(path.resolve()),
    })
    return result


def build_engine_args(
    args: argparse.Namespace,
    weights: list[float],
    metadata: dict[str, Any],
    baseline_checkpoint: Path,
) -> argparse.Namespace:
    device = args.device
    return argparse.Namespace(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        resume=args.resume,
        sample_exposures=EXPOSURES,
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        crop_size=640,
        learning_rate=1e-4,
        weight_decay=0.005,
        validation_every_updates=50_000,
        validation_sample_exposures=list(VALIDATION_MILESTONES),
        checkpoint_every_updates=500,
        max_consecutive_amp_skips=1,
        seed=args.seed,
        device=device,
        amp=True,
        class_weights=weights,
        class_weight_metadata=metadata,
        experiment_mode="class_weighted_CE_20k_resource_adapted_pilot_not_paper_reproduction",
        initial_checkpoint=baseline_checkpoint,
        training_log_filename="training_log.jsonl",
        validation_log_filename="validation_report.jsonl",
        report_filename="experiment_report.json",
        save_final_checkpoint=True,
    )


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    baseline_dir = args.baseline_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    baseline_checkpoint = baseline_dir / "best_model.pth"
    baseline_diagnostics_path = (
        args.baseline_diagnostics.expanduser().resolve()
        if args.baseline_diagnostics
        else output_dir.parent / "standard_ml_unet_180k_diagnostics" / "diagnostic_report.json"
    )
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    if not baseline_checkpoint.is_file():
        raise FileNotFoundError(f"Baseline best checkpoint not found: {baseline_checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty pilot directory: {output_dir}")
    train_ids = engine.read_ids(engine.TRAIN_LIST, "train")
    metadata = scan_train_labels(dataset_root, train_ids)
    metadata["baseline_reference"] = load_baseline_reference(baseline_diagnostics_path)
    weights = metadata["class_weight_vector_id_order"]
    print("Train class frequencies and weights:")
    for name in CLASS_NAMES:
        print(
            f"  {name}: frequency={metadata['class_pixel_frequencies'][name]:.8g}, "
            f"weight={metadata['class_weights'][name]:.6f}"
        )

    engine_args = build_engine_args(args, weights, metadata, baseline_checkpoint)
    reports_before = (
        {path.name for path in output_dir.glob("failure_report*.json")}
        if output_dir.exists()
        else set()
    )
    try:
        report = engine.run(engine_args)
    except Exception as error:
        reports_after = (
            {path.name for path in output_dir.glob("failure_report*.json")}
            if output_dir.exists()
            else set()
        )
        if not (reports_after - reports_before):
            output_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "status": "stopped_with_error",
                "error_type": type(error).__name__,
                "message": str(error),
                "experiment_mode": engine_args.experiment_mode,
                "sample_exposures_target": EXPOSURES,
                "traceback": traceback.format_exc().splitlines(),
            }
            failure_path = output_dir / "failure_report.json"
            index = 2
            while failure_path.exists():
                failure_path = output_dir / f"failure_report_{index}.json"
                index += 1
            failure_path.write_text(
                json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        raise
    report["initialization"] = {
        "source_checkpoint": str(baseline_checkpoint),
        "strict_load": True,
        "initialization_type": "fine_tune_from_existing_best_model",
        "optimizer_state": "new AdamW state initialized for this independent pilot",
    }
    baseline = metadata["baseline_reference"]
    comparisons = []
    for validation in report["training_state"]["validation_history"]:
        current_class_f1 = validation.get("per_class_f1_damage_only", {})
        current_class_precision = validation.get("per_class_precision_damage_only", {})
        current_class_recall = validation.get("per_class_recall_damage_only", {})
        current_class_iou = {
            key: validation.get("per_class_iou", {}).get(key)
            for key in CLASS_NAMES[1:]
        }
        current_metrics = {
            "f1loc": validation.get("f1loc"),
            "f1clf": validation.get("f1clf_harmonic_mean"),
            "oa": validation.get("oa"),
            "miou": validation.get("miou"),
        }
        baseline_metrics = {
            "f1loc": baseline.get("f1loc"),
            "f1clf": baseline.get("f1clf"),
            "oa": baseline.get("oa"),
            "miou": baseline.get("miou"),
        }
        current_class_metrics = {
            "precision": current_class_precision,
            "recall": current_class_recall,
            "f1": current_class_f1,
            "iou": current_class_iou,
        }
        baseline_class_metrics = {
            "precision": baseline.get("per_damage_class_precision") or {},
            "recall": baseline.get("per_damage_class_recall") or {},
            "f1": baseline.get("per_damage_class_f1") or {},
            "iou": baseline.get("per_damage_class_iou") or {},
        }
        comparisons.append(
            {
                "samples_seen": validation.get("samples_seen"),
                "optimizer_update": validation.get("optimizer_update"),
                "current": {
                    **current_metrics,
                    "per_damage_class": current_class_metrics,
                },
                "delta_vs_baseline": {
                    **{
                        name: (current_metrics[name] - baseline_metrics[name])
                        if current_metrics[name] is not None and baseline_metrics[name] is not None
                        else None
                        for name in current_metrics
                    },
                    "per_damage_class": {
                        metric: {
                            name: (
                                current_class_metrics[metric].get(name)
                                - baseline_class_metrics[metric].get(name)
                            )
                            if current_class_metrics[metric].get(name) is not None
                            and baseline_class_metrics[metric].get(name) is not None
                            else None
                            for name in CLASS_NAMES[1:]
                        }
                        for metric in current_class_metrics
                    },
                },
            }
        )
    report["baseline_comparison"] = {
        "baseline": baseline,
        "milestone_comparisons": comparisons,
        "interpretation_priority": [
            "damaged/destroyed recall and F1 become nonzero",
            "official F1clf improves",
            "mIoU improves",
            "OA is supporting context only, not the sole success criterion",
        ],
    }
    report_path = output_dir / "experiment_report.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)
    print("Class-weighted pilot completed:")
    print(f"  output: {output_dir}")
    print(f"  successful updates: {report['training_state']['successful_optimizer_updates']}")
    print(f"  best val mIoU: {report['training_state']['best_val_miou']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
