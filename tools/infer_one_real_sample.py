#!/usr/bin/env python3
"""Run official BRIGHT UNet checkpoint inference on one real TIFF triplet.

The script is a Windows-compatible wrapper around the unmodified official
Dataset, UNet, and Evaluator.  It verifies the published checkpoint bytes,
loads its state dict with strict=True, performs one inference-only forward
pass, saves a raw class-label PNG and an RGB visualization, and reports the
official metrics for this sample/crop as diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from tools.smoke_test_real_data import (  # noqa: E402
    BASELINE_ROOT,
    MultimodalDamageAssessmentDatset,
    UNet,
    array_summary,
    center_crop,
    event_for_sample,
    resolve_device,
    triplet_paths,
    validate_official_loader_result,
    validate_raw_arrays,
)
from util_func.metrics import Evaluator  # noqa: E402


OFFICIAL_CHECKPOINT_FILENAME = "ckpt_UNet_standard_ML_split.pth"
OFFICIAL_CHECKPOINT_SIZE = 124_274_886
OFFICIAL_CHECKPOINT_MD5 = "4dc3dc6995a74faaf37d663e6d200037"
OFFICIAL_CHECKPOINT_URL = (
    "https://zenodo.org/api/records/15349462/files/"
    "ckpt_UNet_standard_ML_split.pth/content"
)
DEFAULT_TEST_LIST = (
    BASELINE_ROOT / "dataset" / "splitname" / "standard_ML" / "test_set.txt"
)
CLASS_NAMES = ("background", "intact", "damaged", "destroyed")
COLOR_MAP = {
    0: (255, 255, 255),
    1: (70, 181, 121),
    2: (228, 189, 139),
    3: (182, 70, 69),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly load the official standard-split UNet checkpoint and infer "
            "one real BRIGHT sample without training or full-set evaluation."
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
        help="Downloaded official standard-split UNet state dict.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=DEFAULT_TEST_LIST,
        help="List from which the one sample must be selected (default: test_set.txt).",
    )
    parser.add_argument(
        "--sample-id",
        help=(
            "Optional exact paper-nine sample ID from --split-file. By default, "
            "the first entry whose TIFF triplet exists is selected."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "one_sample_official_unet",
        help="Directory for prediction PNGs and inference_report.json.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--forward-size",
        type=int,
        default=640,
        help=(
            "Deterministic center-crop size (default: 640; 0 uses the full "
            "1024x1024 sample). It must be divisible by 16."
        ),
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=1024,
        help="Expected raw TIFF height and width (default: 1024; 0 disables).",
    )
    return parser.parse_args()


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {path}")
    sample_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    sample_ids = [sample_id for sample_id in sample_ids if sample_id]
    if not sample_ids:
        raise ValueError(f"Split file is empty: {path}")
    return sample_ids


def select_sample(
    dataset_root: Path,
    split_ids: list[str],
    requested_id: str | None,
) -> tuple[str, str, dict[str, Path]]:
    if requested_id is not None:
        if requested_id not in split_ids:
            raise ValueError(
                f"Requested sample is not present in the selected split: {requested_id}"
            )
        event = event_for_sample(requested_id)
        if event is None:
            raise ValueError(
                f"Requested sample is not in the paper nine-event subset: {requested_id}"
            )
        paths = triplet_paths(dataset_root, requested_id)
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Requested sample is missing one or more TIFFs:\n" + "\n".join(missing)
            )
        return requested_id, event, paths

    candidate_count = 0
    for sample_id in split_ids:
        event = event_for_sample(sample_id)
        if event is None:
            continue
        candidate_count += 1
        paths = triplet_paths(dataset_root, sample_id)
        if all(path.is_file() for path in paths.values()):
            return sample_id, event, paths
    raise FileNotFoundError(
        f"No complete TIFF triplet was found among {candidate_count} paper-nine "
        f"entries in {dataset_root}"
    )


def calculate_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Official checkpoint does not exist: {path}\nDownload it from "
            f"{OFFICIAL_CHECKPOINT_URL}"
        )
    size = path.stat().st_size
    if size != OFFICIAL_CHECKPOINT_SIZE:
        raise ValueError(
            f"Checkpoint size mismatch: got {size}, expected {OFFICIAL_CHECKPOINT_SIZE}"
        )
    md5 = calculate_md5(path)
    if md5.lower() != OFFICIAL_CHECKPOINT_MD5:
        raise ValueError(
            f"Checkpoint MD5 mismatch: got {md5}, expected {OFFICIAL_CHECKPOINT_MD5}"
        )
    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": size,
        "md5": md5,
        "download_url": OFFICIAL_CHECKPOINT_URL,
    }


def load_state_dict_strict(model: torch.nn.Module, path: Path) -> int:
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with PyTorch versions lacking weights_only.
        state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError("Official checkpoint is not a state-dict mapping")
    model.load_state_dict(state_dict, strict=True)
    return len(state_dict)


def finite_number(value: float | np.floating[Any]) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def calculate_official_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    """Reproduce train_UNet.py validation/test metric wiring for one array."""
    target = target.astype(np.int64, copy=False)
    prediction = prediction.astype(np.int64, copy=False)

    target_loc = target.copy()
    target_loc[target_loc > 0] = 1
    prediction_loc = prediction.copy()
    prediction_loc[prediction_loc > 0] = 1

    evaluator_loc = Evaluator(num_class=2)
    evaluator_clf = Evaluator(num_class=4)
    evaluator_total = Evaluator(num_class=4)
    evaluator_loc.add_batch(target_loc, prediction_loc)

    building_mask = target_loc > 0
    evaluator_clf.add_batch(target[building_mask], prediction[building_mask])
    evaluator_total.add_batch(target, prediction)

    with np.errstate(divide="ignore", invalid="ignore"):
        f1loc = evaluator_loc.Pixel_F1_score()
        damage_f1 = evaluator_clf.Damage_F1_score()
        f1clf = len(damage_f1) / np.sum(1.0 / damage_f1)
        oa = evaluator_total.Pixel_Accuracy()
        iou = evaluator_total.Intersection_over_Union()
        miou = evaluator_total.Mean_Intersection_over_Union()

    return {
        "scope": "one sample/crop diagnostic; not a dataset or paper result",
        "value_scale": "fractions in [0,1], not percentages",
        "f1loc": finite_number(f1loc),
        "damage_f1_per_class": {
            name: finite_number(value)
            for name, value in zip(CLASS_NAMES[1:], damage_f1)
        },
        "f1clf_harmonic_mean": finite_number(f1clf),
        "overall_accuracy": finite_number(oa),
        "iou_per_class": {
            name: finite_number(value) for name, value in zip(CLASS_NAMES, iou)
        },
        "mean_iou": finite_number(miou),
        "building_pixels_in_target": int(building_mask.sum()),
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


def colorize(prediction: np.ndarray) -> np.ndarray:
    visualization = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        visualization[prediction == class_id] = color
    return visualization


def save_and_verify_outputs(
    output_dir: Path, sample_id: str, prediction: np.ndarray
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / f"{sample_id}_prediction_label.png"
    color_path = output_dir / f"{sample_id}_prediction_color.png"
    Image.fromarray(prediction.astype(np.uint8)).save(label_path)
    Image.fromarray(colorize(prediction)).save(color_path)

    with Image.open(label_path) as image:
        saved_label = np.asarray(image)
    with Image.open(color_path) as image:
        saved_color = np.asarray(image)
    if not np.array_equal(saved_label, prediction.astype(np.uint8)):
        raise AssertionError("Saved raw prediction PNG does not preserve class IDs")
    if saved_color.shape != (*prediction.shape, 3):
        raise AssertionError(f"Saved color PNG has unexpected shape: {saved_color.shape}")
    return {"label_png": str(label_path), "color_png": str(color_path)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    split_file = args.split_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {dataset_root}")

    checkpoint = verify_official_checkpoint(checkpoint_path)
    split_ids = read_split(split_file)
    sample_id, event, paths = select_sample(dataset_root, split_ids, args.sample_id)

    raw_pre = np.asarray(imageio.imread(paths["pre_event"]))
    raw_post = np.asarray(imageio.imread(paths["post_event"]))
    raw_target = np.asarray(imageio.imread(paths["target"]))
    validate_raw_arrays(raw_pre, raw_post, raw_target, args.expected_size)

    dataset = MultimodalDamageAssessmentDatset(
        str(dataset_root), [sample_id], crop_size=640, max_iters=None, type="test"
    )
    pre, post, loc_target, clf_target, returned_id = dataset[0]
    validate_official_loader_result(
        pre,
        post,
        loc_target,
        clf_target,
        raw_pre,
        raw_post,
        raw_target,
        sample_id,
        returned_id,
    )

    combined = np.concatenate((pre, post), axis=0)
    forward_array, crop = center_crop(combined, args.forward_size)
    top, left = crop["top"], crop["left"]
    height, width = crop["height"], crop["width"]
    target_crop = clf_target[top : top + height, left : left + width]
    input_tensor = torch.from_numpy(np.ascontiguousarray(forward_array)).unsqueeze(0)

    model = UNet(in_channels=6, num_classes=4)
    state_dict_entries = load_state_dict_strict(model, checkpoint_path)
    model.requires_grad_(False)
    model.eval()
    device = resolve_device(args.device)
    model.to(device)
    input_tensor = input_tensor.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            logits = model(input_tensor)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA ran out of memory. Close other GPU applications, retry the "
            "640 crop, or use --device cpu. A smaller multiple of 16 is only a "
            "hardware diagnostic variant."
        ) from exc
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started

    expected_shape = (1, 4, height, width)
    if tuple(logits.shape) != expected_shape:
        raise AssertionError(
            f"Unexpected UNet output shape: {tuple(logits.shape)}; expected {expected_shape}"
        )
    if not torch.isfinite(logits).all():
        raise AssertionError("UNet output contains NaN or infinite values")
    prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    output_files = save_and_verify_outputs(output_dir, sample_id, prediction)
    metrics = calculate_official_metrics(target_crop, prediction)

    device_name = platform.processor() or "CPU"
    peak_memory_mib = None
    total_memory_mib = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_name = properties.name
        peak_memory_mib = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
        total_memory_mib = round(properties.total_memory / 1024**2, 2)

    report_path = output_dir / "inference_report.json"
    report: dict[str, Any] = {
        "status": "passed",
        "scope": (
            "one sample/crop, one strict-checkpoint inference; no training, "
            "parameter update, or full-split evaluation"
        ),
        "checkpoint": {
            **checkpoint,
            "record": "https://zenodo.org/records/15349462",
            "strict_load": True,
            "state_dict_entries": state_dict_entries,
            "training_population_warning": (
                "Official standard_ML checkpoint; not verified as a paper-nine-only "
                "training checkpoint"
            ),
        },
        "sample": {
            "id": sample_id,
            "event": event,
            "split_file": str(split_file),
            "paper_nine_event_member": True,
            "files": {name: str(path) for name, path in paths.items()},
            "raw_pre_event": {**array_summary(raw_pre), "bands": int(raw_pre.shape[2])},
            "raw_post_event": {**array_summary(raw_post), "bands": 1},
            "raw_target": array_summary(raw_target, include_unique=True),
        },
        "inference": {
            "requested_device": args.device,
            "resolved_device": str(device),
            "device_name": device_name,
            "device_total_memory_mib": total_memory_mib,
            "cuda_peak_memory_mib": peak_memory_mib,
            "forward_seconds": round(elapsed_seconds, 6),
            "forward_crop": crop,
            "input_shape": list(input_tensor.shape),
            "input_dtype": str(input_tensor.dtype),
            "output_shape": list(logits.shape),
            "output_dtype": str(logits.dtype),
            "output_class_ids_present": [int(value) for value in np.unique(prediction)],
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "outputs": {**output_files, "report_json": str(report_path)},
        "official_evaluator_diagnostic": metrics,
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "imageio": __import__("imageio").__version__,
            "pillow": __import__("PIL").__version__,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
