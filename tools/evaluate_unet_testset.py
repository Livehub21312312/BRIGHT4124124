#!/usr/bin/env python3
"""Evaluate the unmodified BRIGHT UNet on full-resolution test TIFFs.

This Windows-compatible wrapper preserves the official Dataset, UNet, and
Evaluator behavior while adding strict checkpoint loading, paper-nine list
filtering, reproducibility metadata, prediction files, and failure reporting.
It intentionally performs whole-image 1024x1024 inference with batch size one:
there is no crop, sliding window, resize, TTA, autocast, or backward pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from tools.infer_one_real_sample import (  # noqa: E402
    CLASS_NAMES,
    COLOR_MAP,
    OFFICIAL_CHECKPOINT_FILENAME,
    OFFICIAL_CHECKPOINT_MD5,
    OFFICIAL_CHECKPOINT_SIZE,
    OFFICIAL_CHECKPOINT_URL,
    calculate_md5,
    finite_number,
    load_state_dict_strict,
)
from tools.smoke_test_real_data import (  # noqa: E402
    BASELINE_ROOT,
    PAPER_EVENTS,
    MultimodalDamageAssessmentDatset,
    UNet,
    event_for_sample,
    resolve_device,
)
from util_func.metrics import Evaluator  # noqa: E402


DEFAULT_TEST_LIST = (
    BASELINE_ROOT / "dataset" / "splitname" / "standard_ML" / "test_set.txt"
)
EXPECTED_PAPER_NINE_TEST_COUNT = 661


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BRIGHT UNet with the official whole-image Dataset and "
            "Evaluator wiring."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="BRIGHT root containing pre-event, post-event, and target folders.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / OFFICIAL_CHECKPOINT_FILENAME,
        help="Raw UNet state dict; it is always loaded with strict=True.",
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        default=DEFAULT_TEST_LIST,
        help="Test sample-ID list (default: official standard_ML/test_set.txt).",
    )
    parser.add_argument(
        "--paper-nine-only",
        action="store_true",
        help=(
            "Keep only the nine paper events and require exactly 661 selected "
            "samples. The source list is never modified."
        ),
    )
    parser.add_argument(
        "--expected-checkpoint-md5",
        help="Optional lowercase/uppercase MD5 that must match before model loading.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "unet_testset_evaluation",
        help="Directory for predictions, manifests, failures, and metrics JSON.",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=1024,
        help="Required square height/width for every loaded sample (default: 1024).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress after this many attempted samples; 0 disables.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Validate/filter the list and write the 661-ID manifest, then stop "
            "without reading data or loading the checkpoint."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_test_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Test list does not exist: {path}")
    sample_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    sample_ids = [sample_id for sample_id in sample_ids if sample_id]
    if not sample_ids:
        raise ValueError(f"Test list is empty: {path}")
    duplicates = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Test list contains duplicate sample IDs: {duplicates[:10]}")
    return sample_ids


def select_test_ids(all_ids: list[str], paper_nine_only: bool) -> list[str]:
    if not paper_nine_only:
        return list(all_ids)
    selected = [sample_id for sample_id in all_ids if event_for_sample(sample_id)]
    if len(selected) != EXPECTED_PAPER_NINE_TEST_COUNT:
        raise ValueError(
            "Paper-nine filter did not produce the required 661 samples: "
            f"got {len(selected)}"
        )
    return selected


def event_counts(sample_ids: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sample_id in sample_ids:
        counts[event_for_sample(sample_id) or "outside-paper-nine"] += 1
    ordered = {event: counts[event] for event in PAPER_EVENTS if counts[event]}
    if counts["outside-paper-nine"]:
        ordered["outside-paper-nine"] = counts["outside-paper-nine"]
    return ordered


def checkpoint_metadata(path: Path, expected_md5: str | None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    size = path.stat().st_size
    md5 = calculate_md5(path).lower()
    if expected_md5 is not None and md5 != expected_md5.lower():
        raise ValueError(
            f"Checkpoint MD5 mismatch: got {md5}, expected {expected_md5.lower()}"
        )
    is_official_standard = (
        size == OFFICIAL_CHECKPOINT_SIZE and md5 == OFFICIAL_CHECKPOINT_MD5
    )
    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": size,
        "md5": md5,
        "expected_md5_argument": expected_md5.lower() if expected_md5 else None,
        "matches_published_standard_ml_file": is_official_standard,
        "published_standard_ml_filename": OFFICIAL_CHECKPOINT_FILENAME,
        "published_standard_ml_size_bytes": OFFICIAL_CHECKPOINT_SIZE,
        "published_standard_ml_md5": OFFICIAL_CHECKPOINT_MD5,
        "published_download_url": OFFICIAL_CHECKPOINT_URL,
        "strict_load": True,
        "training_scope_warning": (
            "The published checkpoint is labeled standard_ML. Its exact training-ID "
            "manifest is not supplied, so it is not verified as paper-nine-only."
            if is_official_standard
            else "Training population is not inferred from this checkpoint file."
        ),
    }


def save_prediction(
    original_dir: Path,
    colored_dir: Path,
    sample_id: str,
    prediction: np.ndarray,
) -> None:
    Image.fromarray(prediction.astype(np.uint8)).save(
        original_dir / f"{sample_id}_building_damage.png"
    )
    visualization = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        visualization[prediction == class_id] = color
    Image.fromarray(visualization).save(
        colored_dir / f"{sample_id}_building_damage.png"
    )


def official_aggregate_metrics(
    evaluator_loc: Evaluator,
    evaluator_clf: Evaluator,
    evaluator_total: Evaluator,
) -> dict[str, Any]:
    """Call the same Evaluator methods and harmonic wiring as train_UNet.py."""
    with np.errstate(divide="ignore", invalid="ignore"):
        loc_precision = evaluator_loc.Pixel_Precision_Rate()
        loc_recall = evaluator_loc.Pixel_Recall_Rate()
        f1loc = evaluator_loc.Pixel_F1_score()
        damage_f1 = evaluator_clf.Damage_F1_score()
        f1clf = len(damage_f1) / np.sum(1.0 / damage_f1)
        oa = evaluator_total.Pixel_Accuracy()
        iou = evaluator_total.Intersection_over_Union()
        miou = evaluator_total.Mean_Intersection_over_Union()

    return {
        "value_scale": "fractions in [0,1], not percentages",
        "f1loc": finite_number(f1loc),
        "localization_precision": finite_number(loc_precision),
        "localization_recall": finite_number(loc_recall),
        "f1clf_harmonic_mean": finite_number(f1clf),
        "per_class_f1": {
            "background": None,
            "intact": finite_number(damage_f1[0]),
            "damaged": finite_number(damage_f1[1]),
            "destroyed": finite_number(damage_f1[2]),
        },
        "per_class_f1_note": (
            "Official Damage_F1_score excludes class 0; background F1 is not "
            "defined by the official Evaluator and is not reimplemented here."
        ),
        "overall_accuracy": finite_number(oa),
        "per_class_iou": {
            name: finite_number(value) for name, value in zip(CLASS_NAMES, iou)
        },
        "mean_iou": finite_number(miou),
        "confusion_matrix_localization_rows_gt_cols_prediction": (
            evaluator_loc.confusion_matrix.astype(np.int64).tolist()
        ),
        "confusion_matrix_damage_on_gt_building_pixels": (
            evaluator_clf.confusion_matrix.astype(np.int64).tolist()
        ),
        "confusion_matrix_four_class_all_pixels": (
            evaluator_total.confusion_matrix.astype(np.int64).tolist()
        ),
    }


def write_failures(path: Path, failures: list[dict[str, str]]) -> None:
    fields = ("sample_id", "event", "error_type", "message")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare_manifest(args: argparse.Namespace) -> tuple[Path, Path, list[str], dict[str, Any]]:
    test_list = args.test_list.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_ids = read_test_list(test_list)
    selected_ids = select_test_ids(all_ids, args.paper_nine_only)
    manifest_path = output_dir / "evaluated_samples.txt"
    manifest_path.write_text(
        "".join(f"{sample_id}\n" for sample_id in selected_ids), encoding="utf-8"
    )
    selection = {
        "source_test_list": str(test_list),
        "source_test_list_sha256": sha256_file(test_list),
        "source_nonempty_sample_count": len(all_ids),
        "paper_nine_only": args.paper_nine_only,
        "selected_manifest": str(manifest_path),
        "selected_manifest_sha256": sha256_lines(selected_ids),
        "selected_sample_count": len(selected_ids),
        "selected_unique_sample_count": len(set(selected_ids)),
        "selected_event_counts": event_counts(selected_ids),
        "paper_events": list(PAPER_EVENTS),
    }
    return test_list, output_dir, selected_ids, selection


def evaluate(args: argparse.Namespace) -> int:
    test_list, output_dir, selected_ids, selection = prepare_manifest(args)
    if args.prepare_only:
        plan = {
            "status": "prepared",
            "scope": "list filtering only; no TIFF read, checkpoint load, or inference",
            "selection": selection,
        }
        write_json(output_dir / "evaluation_plan.json", plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {dataset_root}")
    if args.expected_size <= 0 or args.expected_size % 16:
        raise ValueError("--expected-size must be a positive multiple of 16")

    checkpoint = checkpoint_metadata(checkpoint_path, args.expected_checkpoint_md5)
    model = UNet(in_channels=6, num_classes=4)
    state_dict_entries = load_state_dict_strict(model, checkpoint_path)
    model.requires_grad_(False)
    model.eval()
    device = resolve_device(args.device)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    dataset = MultimodalDamageAssessmentDatset(
        str(dataset_root), selected_ids, crop_size=1024, max_iters=None, type="test"
    )
    evaluator_loc = Evaluator(num_class=2)
    evaluator_clf = Evaluator(num_class=4)
    evaluator_total = Evaluator(num_class=4)
    original_dir = output_dir / "original"
    colored_dir = output_dir / "colored"
    original_dir.mkdir(parents=True, exist_ok=True)
    colored_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, str]] = []
    completed_ids: list[str] = []
    successful_events: Counter[str] = Counter()
    output_class_ids: set[int] = set()
    inference_seconds = 0.0
    started_total = time.perf_counter()

    for index, sample_id in enumerate(selected_ids):
        event = event_for_sample(sample_id) or "outside-paper-nine"
        try:
            pre, post, labels_loc, labels_clf, returned_id = dataset[index]
            if returned_id != sample_id:
                raise AssertionError(
                    f"Dataset returned {returned_id!r}, expected {sample_id!r}"
                )
            expected_chw = (3, args.expected_size, args.expected_size)
            expected_hw = (args.expected_size, args.expected_size)
            if pre.shape != expected_chw or post.shape != expected_chw:
                raise ValueError(
                    f"Expected pre/post {expected_chw}, got {pre.shape}/{post.shape}"
                )
            if labels_loc.shape != expected_hw or labels_clf.shape != expected_hw:
                raise ValueError(
                    "Expected full-size labels "
                    f"{expected_hw}, got {labels_loc.shape}/{labels_clf.shape}"
                )

            input_array = np.ascontiguousarray(np.concatenate((pre, post), axis=0))
            input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started_inference = time.perf_counter()
            with torch.inference_mode():
                logits = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started_inference

            expected_output = (1, 4, args.expected_size, args.expected_size)
            if tuple(logits.shape) != expected_output:
                raise AssertionError(
                    f"Expected output {expected_output}, got {tuple(logits.shape)}"
                )
            if not torch.isfinite(logits).all():
                raise ValueError("Model output contains NaN or infinite values")
            prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            save_prediction(original_dir, colored_dir, sample_id, prediction)
            output_loc = prediction.copy()
            output_loc[output_loc > 0] = 1
            evaluator_loc.add_batch(labels_loc, output_loc)
            building_mask = labels_loc > 0
            evaluator_clf.add_batch(
                labels_clf[building_mask], prediction[building_mask]
            )
            evaluator_total.add_batch(labels_clf, prediction)
            output_class_ids.update(int(value) for value in np.unique(prediction))
            completed_ids.append(sample_id)
            successful_events[event] += 1
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA out of memory during official full 1024x1024 inference. "
                "Do not replace it with a 640 crop or sliding window for the strict "
                "evaluation; close GPU applications or use --device cpu."
            ) from exc
        except Exception as exc:  # Continue to produce the requested failure list.
            failures.append(
                {
                    "sample_id": sample_id,
                    "event": event,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        finally:
            if args.progress_every and (
                (index + 1) % args.progress_every == 0
                or index + 1 == len(selected_ids)
            ):
                print(
                    f"attempted={index + 1}/{len(selected_ids)} "
                    f"completed={len(completed_ids)} failed={len(failures)}",
                    flush=True,
                )

    elapsed_total = time.perf_counter() - started_total
    if not completed_ids:
        raise RuntimeError("No samples completed; aggregate metrics are unavailable")

    completed_path = output_dir / "completed_samples.txt"
    completed_path.write_text(
        "".join(f"{sample_id}\n" for sample_id in completed_ids), encoding="utf-8"
    )
    failure_path = output_dir / "failed_samples.csv"
    write_failures(failure_path, failures)
    metrics = official_aggregate_metrics(
        evaluator_loc, evaluator_clf, evaluator_total
    )

    device_name = platform.processor() or "CPU"
    peak_memory_mib = None
    total_memory_mib = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_name = properties.name
        peak_memory_mib = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
        total_memory_mib = round(properties.total_memory / 1024**2, 2)

    status = "passed" if not failures else "partial"
    report_path = output_dir / "evaluation_report.json"
    report: dict[str, Any] = {
        "status": status,
        "scope": (
            "official whole-image evaluation over the selected manifest; no "
            "training, crop, sliding window, resize, TTA, or parameter update"
        ),
        "parameters": {
            "dataset_root": str(dataset_root),
            "checkpoint": str(checkpoint_path),
            "test_list": str(test_list),
            "paper_nine_only": args.paper_nine_only,
            "expected_checkpoint_md5": args.expected_checkpoint_md5,
            "device": args.device,
            "output_dir": str(output_dir),
            "expected_size": args.expected_size,
            "progress_every": args.progress_every,
            "prepare_only": False,
        },
        "checkpoint": {**checkpoint, "state_dict_entries": state_dict_entries},
        "selection": selection,
        "execution": {
            "dataset_root": str(dataset_root),
            "test_list": str(test_list),
            "attempted_samples": len(selected_ids),
            "completed_samples": len(completed_ids),
            "failed_samples": len(failures),
            "completed_event_counts": dict(successful_events),
            "batch_size": 1,
            "input_shape_per_sample": [1, 6, args.expected_size, args.expected_size],
            "output_shape_per_sample": [1, 4, args.expected_size, args.expected_size],
            "output_class_ids_present": sorted(output_class_ids),
            "input_dtype": "torch.float32",
            "model_mode": "eval",
            "inference_mode": True,
            "tta": False,
            "whole_image": True,
            "sliding_window": False,
            "crop": None,
            "resize": None,
            "requested_device": args.device,
            "resolved_device": str(device),
            "device_name": device_name,
            "device_total_memory_mib": total_memory_mib,
            "cuda_peak_memory_mib": peak_memory_mib,
            "model_forward_seconds_total": round(inference_seconds, 6),
            "wall_seconds_total": round(elapsed_total, 6),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "official_metrics": {
            "coverage": (
                f"aggregate confusion matrices over {len(completed_ids)} successful "
                f"samples out of {len(selected_ids)} selected"
            ),
            "complete_selected_manifest": not failures,
            **metrics,
        },
        "failures": failures,
        "artifacts": {
            "raw_predictions_dir": str(original_dir),
            "colored_predictions_dir": str(colored_dir),
            "selected_manifest": selection["selected_manifest"],
            "completed_manifest": str(completed_path),
            "failure_csv": str(failure_path),
            "report_json": str(report_path),
        },
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": __import__("PIL").__version__,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if not failures else 1


def main() -> None:
    raise SystemExit(evaluate(parse_args()))


if __name__ == "__main__":
    main()
