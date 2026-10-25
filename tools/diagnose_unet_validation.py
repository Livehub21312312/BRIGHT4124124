#!/usr/bin/env python3
"""Read-only diagnostics for a BRIGHT UNet checkpoint on standard_ML val only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "bda_benchmark"
sys.path.insert(0, str(BASELINE))
sys.dont_write_bytecode = True

from dataset.make_data_loader import MultimodalDamageAssessmentDatset  # noqa: E402
from model.UNet import UNet  # noqa: E402
from util_func.metrics import Evaluator  # noqa: E402


VAL_LIST = BASELINE / "dataset" / "splitname" / "standard_ML" / "val_set.txt"
CLASS_NAMES = ("background", "intact", "damaged", "destroyed")
COLORS = np.asarray(
    ((255, 255, 255), (70, 181, 121), (228, 189, 139), (182, 70, 69)),
    dtype=np.uint8,
)
MEAN = np.asarray((123.675, 116.28, 103.53), dtype=np.float32)
STD = np.asarray((58.395, 57.12, 57.375), dtype=np.float32)
EXPECTED_VAL_COUNT = 450


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose a saved BRIGHT UNet on official standard_ML val only."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "standard_ml_unet_180k_resource_adapted" / "best_model.pth",
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=ROOT / "outputs" / "standard_ml_unet_180k_resource_adapted" / "training_report.json",
    )
    parser.add_argument(
        "--val-list", type=Path, default=VAL_LIST,
        help="Official standard_ML val_set.txt; no test-list option is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "standard_ml_unet_180k_diagnostics",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--representative-count", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_for(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation list not found: {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [sample_id for sample_id in ids if sample_id]
    if len(ids) != EXPECTED_VAL_COUNT or len(set(ids)) != len(ids):
        raise ValueError(
            f"Expected {EXPECTED_VAL_COUNT} unique official validation IDs, got {len(ids)}"
        )
    return ids


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but it is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inspect_training_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path), "reason": "file not found"}
    report = json.loads(path.read_text(encoding="utf-8"))
    state = report.get("training_state", {})
    history = state.get("validation_history", [])
    valid = [entry for entry in history if entry.get("miou") is not None]
    if not valid:
        return {
            "available": True,
            "path": str(path),
            "status": report.get("status"),
            "reason": "no validation_history entries with miou found",
            "best_validation": None,
            "final_validation": history[-1] if history else None,
        }
    max_miou = max(float(entry["miou"]) for entry in valid)
    best = next(entry for entry in valid if float(entry["miou"]) == max_miou)
    return {
        "available": True,
        "path": str(path),
        "status": report.get("status"),
        "experiment_mode": report.get("experiment_mode")
        or report.get("configuration", {}).get("experiment_mode"),
        "paper_reproduction_status": report.get("paper_reproduction_status")
        or report.get("configuration", {}).get("paper_reproduction_status"),
        "sample_exposures": report.get("configuration", {}).get("sample_exposures"),
        "successful_optimizer_updates": state.get("successful_optimizer_updates"),
        "optimizer_update_attempts": state.get("optimizer_update_attempts"),
        "best_validation": best,
        "best_validation_update": best.get("optimizer_update"),
        "best_validation_miou": max_miou,
        "reported_best_update": state.get("best_update"),
        "reported_best_miou": state.get("best_val_miou"),
        "final_validation": history[-1] if history else None,
    }


def load_raw_state_dict(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a state-dict mapping; got {type(payload).__name__}")
    if payload and all(isinstance(key, str) for key in payload) and "state_dict" in payload:
        raise TypeError("Checkpoint is wrapped; expected best_model.pth raw state_dict format")
    if not all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise TypeError("Checkpoint contains non-tensor values; expected raw state_dict")
    model.load_state_dict(payload, strict=True)
    return payload


def counts_for(array: np.ndarray) -> dict[str, Any]:
    valid = (array >= 0) & (array < len(CLASS_NAMES))
    counts = np.bincount(array[valid].astype(np.int64), minlength=4)
    total_valid = int(counts.sum())
    total_pixels = int(array.size)
    return {
        "class_pixel_counts": {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)},
        "class_pixel_percent_valid": {
            name: (float(counts[i] / total_valid * 100) if total_valid else None)
            for i, name in enumerate(CLASS_NAMES)
        },
        "valid_pixel_count": total_valid,
        "ignored_or_invalid_pixel_count": total_pixels - total_valid,
        "total_pixel_count": total_pixels,
    }


def finite(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def official_metrics(
    eval_loc: Evaluator, eval_clf: Evaluator, eval_all: Evaluator
) -> dict[str, Any]:
    with np.errstate(divide="ignore", invalid="ignore"):
        f1loc = eval_loc.Pixel_F1_score()
        damage_f1 = eval_clf.Damage_F1_score()
        f1clf = len(damage_f1) / np.sum(1.0 / damage_f1)
        oa = eval_all.Pixel_Accuracy()
        per_iou = eval_all.Intersection_over_Union()
        miou = eval_all.Mean_Intersection_over_Union()
        precision = eval_clf.calculate_per_class_metrics()[0] / (
            eval_clf.calculate_per_class_metrics()[0]
            + eval_clf.calculate_per_class_metrics()[2]
            + 1e-7
        )
        recall = eval_clf.calculate_per_class_metrics()[0] / (
            eval_clf.calculate_per_class_metrics()[0]
            + eval_clf.calculate_per_class_metrics()[1]
            + 1e-7
        )
    return {
        "metric_scale": "fraction [0,1]",
        "f1loc": finite(f1loc),
        "f1clf_official_harmonic_mean_damage_classes_1_to_3": finite(f1clf),
        "per_damage_class_f1": {
            name: finite(value)
            for name, value in zip(CLASS_NAMES[1:], damage_f1)
        },
        "per_damage_class_precision": {
            name: finite(value) for name, value in zip(CLASS_NAMES[1:], precision)
        },
        "per_damage_class_recall": {
            name: finite(value) for name, value in zip(CLASS_NAMES[1:], recall)
        },
        "oa": finite(oa),
        "miou": finite(miou),
        "per_class_iou": {
            name: finite(value) for name, value in zip(CLASS_NAMES, per_iou)
        },
        "confusion_matrix_four_class_all_pixels_rows_gt_cols_prediction": (
            eval_all.confusion_matrix.astype(np.int64).tolist()
        ),
        "confusion_matrix_damage_evaluator_gt_building_pixels_rows_gt_cols_prediction": (
            eval_clf.confusion_matrix.astype(np.int64).tolist()
        ),
        "confusion_matrix_localization_rows_gt_cols_prediction": (
            eval_loc.confusion_matrix.astype(np.int64).tolist()
        ),
        "f1clf_definition": (
            "official Evaluator Damage_F1_score for classes 1,2,3 on pixels where "
            "ground-truth localization label > 0, combined by harmonic mean"
        ),
    }


def to_rgb(normalized_chw: np.ndarray) -> np.ndarray:
    hwc = np.transpose(normalized_chw, (1, 2, 0))
    raw = np.clip(hwc * STD + MEAN, 0, 255).astype(np.uint8)
    return raw


def colorize(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = (labels >= 0) & (labels < 4)
    out[valid] = COLORS[labels[valid].astype(np.int64)]
    out[~valid] = (0, 0, 0)
    return out


def make_candidate(
    sample_id: str,
    event: str,
    pre: np.ndarray,
    post: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    images = [
        Image.fromarray(to_rgb(pre)),
        Image.fromarray(to_rgb(post)),
        Image.fromarray(colorize(target)),
        Image.fromarray(colorize(prediction)),
    ]
    size = (384, 384)
    thumbs = [np.asarray(image.resize(size, Image.Resampling.NEAREST)) for image in images]
    gt = np.bincount(target[(target >= 0) & (target < 4)].astype(np.int64), minlength=4)
    pred = np.bincount(prediction[(prediction >= 0) & (prediction < 4)].astype(np.int64), minlength=4)
    score = (
        int(gt[2] > 0) + int(gt[3] > 0),
        int(gt[3]),
        int(gt[2]),
        int(pred[2] > 0) + int(pred[3] > 0),
    )
    return {
        "sample_id": sample_id,
        "event": event,
        "score": score,
        "images": thumbs,
        "ground_truth_counts": gt,
        "prediction_counts": pred,
    }


def write_candidate(path: Path, candidate: dict[str, Any]) -> None:
    labels = ("Pre-event optical", "Post-event SAR", "Ground truth", "Prediction")
    canvas = Image.new("RGB", (4 * 384, 384 + 36), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(zip(labels, candidate["images"])):
        canvas.paste(Image.fromarray(image), (index * 384, 36))
        draw.text((index * 384 + 8, 10), title, fill=(255, 255, 255))
    canvas.save(path)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty diagnostic directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    report_path = args.training_report.expanduser().resolve()
    val_path = args.val_list.expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    for path in (checkpoint_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.representative_count < 1 or args.max_candidates < args.representative_count:
        raise ValueError("Require representative-count >= 1 and max-candidates >= it")

    ids = read_ids(val_path)
    device = choose_device(args.device)
    model = UNet(in_channels=6, num_classes=4)
    state_dict = load_raw_state_dict(model, checkpoint_path)
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    dataset = MultimodalDamageAssessmentDatset(
        str(dataset_root), ids, crop_size=1024, max_iters=None, type="test"
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    eval_loc, eval_clf, eval_all = Evaluator(2), Evaluator(4), Evaluator(4)
    target_counts = np.zeros(4, dtype=np.int64)
    prediction_counts = np.zeros(4, dtype=np.int64)
    ignored_target_pixels = 0
    per_sample: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (pre, post, labels_loc, labels_clf, sample_ids) in enumerate(loader):
        sample_id = sample_ids[0]
        pre_np = pre.squeeze(0).numpy()
        post_np = post.squeeze(0).numpy()
        target = labels_clf.squeeze(0).numpy().astype(np.int64, copy=False)
        loc = labels_loc.squeeze(0).numpy().astype(np.int64, copy=False)
        inputs = torch.cat((pre, post), dim=1).to(device)
        with torch.inference_mode():
            logits = model(inputs)
        if logits.shape != (1, 4, 1024, 1024) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError(f"Invalid model output for {sample_id}: {tuple(logits.shape)}")
        prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
        if not np.isin(target, (0, 1, 2, 3, 255)).all():
            bad = np.unique(target[~np.isin(target, (0, 1, 2, 3, 255))]).tolist()
            raise ValueError(f"Unexpected label value(s) for {sample_id}: {bad}")

        valid = (target >= 0) & (target < 4)
        target_counts += np.bincount(target[valid], minlength=4)
        prediction_counts += np.bincount(prediction[valid], minlength=4)
        ignored_target_pixels += int((~valid).sum())
        output_loc = prediction.copy()
        output_loc[output_loc > 0] = 1
        eval_loc.add_batch(loc, output_loc)
        building_mask = loc > 0
        eval_clf.add_batch(target[building_mask], prediction[building_mask])
        eval_all.add_batch(target, prediction)

        row: dict[str, Any] = {
            "sample_id": sample_id,
            "event": event_for(sample_id),
        }
        for label, array in (("gt", target), ("pred", prediction)):
            counts = np.bincount(array[(array >= 0) & (array < 4)], minlength=4)
            for class_id, class_name in enumerate(CLASS_NAMES):
                row[f"{label}_{class_name}_pixels"] = int(counts[class_id])
        per_sample.append(row)

        gt_counts = np.bincount(target[valid], minlength=4)
        if gt_counts[2] or gt_counts[3]:
            candidate = make_candidate(
                sample_id, event_for(sample_id), pre_np, post_np, target, prediction
            )
            candidates.append(candidate)
            candidates.sort(key=lambda item: item["score"], reverse=True)
            del candidates[args.max_candidates :]
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(f"validated {index + 1}/{len(ids)}", flush=True)

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    viz_dir = output / "representative_samples"
    viz_dir.mkdir(parents=True, exist_ok=True)
    selected = candidates[: args.representative_count]
    for candidate in selected:
        write_candidate(viz_dir / f"{candidate['sample_id']}_diagnostic.png", candidate)

    metrics = official_metrics(eval_loc, eval_clf, eval_all)
    total_valid = int(target_counts.sum())
    predictions_over_valid = {
        name: int(prediction_counts[i]) for i, name in enumerate(CLASS_NAMES)
    }
    target_over_valid = {name: int(target_counts[i]) for i, name in enumerate(CLASS_NAMES)}
    ratios = {
        name: float(target_counts[i] / total_valid * 100) if total_valid else None
        for i, name in enumerate(CLASS_NAMES)
    }
    pred_ratios = {
        name: float(prediction_counts[i] / total_valid * 100) if total_valid else None
        for i, name in enumerate(CLASS_NAMES)
    }
    if report_path.is_file():
        training_summary = inspect_training_report(report_path)
    else:
        training_summary = {
            "available": False,
            "path": str(report_path),
            "reason": "training report not present in this execution environment",
        }
    summary = {
        "status": "completed",
        "scope": "official standard_ML validation split only; no train or test images read",
        "sample_count": len(ids),
        "successful_sample_count": len(per_sample),
        "failed_sample_count": 0,
        "dataset_root": str(dataset_root),
        "validation_list": str(val_path),
        "validation_list_sha256": sha256_file(val_path),
        "checkpoint": {
            "path": str(checkpoint_path),
            "filename": checkpoint_path.name,
            "format": "raw state_dict mapping[str, Tensor]",
            "strict_load": True,
            "state_dict_tensor_count": len(state_dict),
            "state_dict_key_count": len(state_dict),
            "parameter_shapes": {key: list(value.shape) for key, value in state_dict.items()},
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
        "training_report_review": training_summary,
        "device": str(device),
        "class_id_mapping": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "ground_truth_distribution": {
            "counts": target_over_valid,
            "percent_valid_pixels": ratios,
            "ignored_or_invalid_pixels": ignored_target_pixels,
            "valid_pixels": total_valid,
        },
        "prediction_distribution_on_valid_ground_truth_pixels": {
            "counts": predictions_over_valid,
            "percent_valid_pixels": pred_ratios,
        },
        "official_metrics": metrics,
        "per_sample_csv": str(output / "per_sample_class_counts.csv"),
        "representative_visualizations": [
            {
                "sample_id": item["sample_id"],
                "event": item["event"],
                "path": str(viz_dir / f"{item['sample_id']}_diagnostic.png"),
                "ground_truth_class_counts": {
                    name: int(item["ground_truth_counts"][i])
                    for i, name in enumerate(CLASS_NAMES)
                },
                "prediction_class_counts": {
                    name: int(item["prediction_counts"][i])
                    for i, name in enumerate(CLASS_NAMES)
                },
            }
            for item in selected
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    with (output / "per_sample_class_counts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(per_sample[0]) if per_sample else ["sample_id", "event"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_sample)
    write_json(output / "diagnostic_report.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Diagnostic report: {args.output_dir / 'diagnostic_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
