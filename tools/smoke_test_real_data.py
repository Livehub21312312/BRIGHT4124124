#!/usr/bin/env python3
"""Run one read-only BRIGHT real-data forward-pass smoke test.

This wrapper deliberately imports the official Dataset and UNet implementations
without changing them.  It selects one available sample from the nine-event
paper subset of the official training list, validates the raw TIFF triplet,
loads it through the official Dataset, and performs exactly one inference-mode
forward pass.  It never performs backward propagation or updates parameters.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "bda_benchmark"
DEFAULT_TRAIN_LIST = (
    BASELINE_ROOT / "dataset" / "splitname" / "standard_ML" / "train_set.txt"
)
PAPER_EVENTS = (
    "bata-explosion",
    "beirut-explosion",
    "congo-volcano",
    "haiti-earthquake",
    "hawaii-wildfire",
    "la_palma-volcano",
    "libya-flood",
    "morocco-earthquake",
    "turkey-earthquake",
)
ALLOWED_LABELS = {0, 1, 2, 3}
NORMALIZE_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
NORMALIZE_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)

# The official loader imports matplotlib and writes a cache on first import.
# Keep that cache outside the repository and make this work on Windows too.
CACHE_ROOT = Path(tempfile.gettempdir()) / "bright-reproduction-cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))
sys.dont_write_bytecode = True

from dataset.make_data_loader import MultimodalDamageAssessmentDatset  # noqa: E402
from model.UNet import UNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one real BRIGHT training sample with the official loader and "
            "run one inference-only UNet forward pass."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="BRIGHT root containing pre-event, post-event, and target folders.",
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=DEFAULT_TRAIN_LIST,
        help="Official training split file (default: repository standard_ML list).",
    )
    parser.add_argument(
        "--sample-id",
        help=(
            "Optional exact sample ID. It must occur in the training list and "
            "belong to the paper's nine-event subset. By default, the first "
            "sample whose three TIFFs exist is selected."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Forward-pass device. auto uses CUDA when PyTorch can access it.",
    )
    parser.add_argument(
        "--forward-size",
        type=int,
        default=640,
        help=(
            "Deterministic center-crop size used only for the forward pass "
            "(default: official training crop 640; 0 uses the full image)."
        ),
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=1024,
        help="Expected raw TIFF height and width (default: 1024; 0 disables).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Optional compatible UNet state dict, such as the official "
            "ckpt_UNet_standard_ML_split.pth. Randomly initialized weights are "
            "used when this option is omitted."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path at which to save the same JSON printed to stdout.",
    )
    return parser.parse_args()


def event_for_sample(sample_id: str) -> str | None:
    for event in PAPER_EVENTS:
        if sample_id.startswith(f"{event}_"):
            return event
    return None


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


def read_train_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Training split file does not exist: {path}")
    sample_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    sample_ids = [sample_id for sample_id in sample_ids if sample_id]
    if not sample_ids:
        raise ValueError(f"Training split file is empty: {path}")
    return sample_ids


def select_sample(
    dataset_root: Path, train_ids: list[str], requested_id: str | None
) -> tuple[str, str, dict[str, Path]]:
    if requested_id is not None:
        if requested_id not in train_ids:
            raise ValueError(
                f"Requested sample is not in the selected training list: {requested_id}"
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

    paper_ids = [sample_id for sample_id in train_ids if event_for_sample(sample_id)]
    for sample_id in paper_ids:
        paths = triplet_paths(dataset_root, sample_id)
        if all(path.is_file() for path in paths.values()):
            event = event_for_sample(sample_id)
            assert event is not None
            return sample_id, event, paths

    raise FileNotFoundError(
        f"No complete TIFF triplet was found for the {len(paper_ids)} paper-nine "
        f"samples in {dataset_root}"
    )


def array_summary(array: np.ndarray, *, include_unique: bool = False) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "all_finite": bool(np.isfinite(array).all()),
    }
    if include_unique:
        summary["unique_values"] = [
            int(value) if float(value).is_integer() else float(value)
            for value in np.unique(array)
        ]
    return summary


def validate_raw_arrays(
    pre: np.ndarray,
    post: np.ndarray,
    label: np.ndarray,
    expected_size: int,
) -> None:
    if pre.ndim != 3 or pre.shape[2] < 3:
        raise ValueError(
            f"Pre-event TIFF must be HxWx>=3, but its shape is {pre.shape}"
        )
    if post.ndim != 2:
        raise ValueError(
            "Post-event TIFF must be a two-dimensional single-band array for "
            f"the official np.stack call, but its shape is {post.shape}"
        )
    if label.ndim != 2:
        raise ValueError(f"Target TIFF must be two-dimensional, but its shape is {label.shape}")

    spatial_shapes = {tuple(pre.shape[:2]), tuple(post.shape), tuple(label.shape)}
    if len(spatial_shapes) != 1:
        raise ValueError(
            f"Spatial dimensions do not match: pre={pre.shape}, post={post.shape}, "
            f"target={label.shape}"
        )
    if expected_size and tuple(label.shape) != (expected_size, expected_size):
        raise ValueError(
            f"Expected {expected_size}x{expected_size} TIFFs, found {label.shape}"
        )

    for name, array in (("pre-event", pre), ("post-event", post), ("target", label)):
        if array.size == 0:
            raise ValueError(f"{name} TIFF is empty")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} TIFF is not numeric: {array.dtype}")
        if np.issubdtype(array.dtype, np.complexfloating):
            raise TypeError(f"{name} TIFF has an unsupported complex dtype: {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} TIFF contains NaN or infinite values")

    if not np.all(label == np.floor(label)):
        raise ValueError("Target TIFF contains non-integer label values")
    label_values = {int(value) for value in np.unique(label)}
    invalid_labels = sorted(label_values - ALLOWED_LABELS)
    if invalid_labels:
        raise ValueError(f"Target TIFF contains invalid labels: {invalid_labels}")


def validate_official_loader_result(
    pre: np.ndarray,
    post: np.ndarray,
    loc_label: np.ndarray,
    clf_label: np.ndarray,
    raw_pre: np.ndarray,
    raw_post: np.ndarray,
    raw_label: np.ndarray,
    sample_id: str,
    returned_id: str,
) -> None:
    height, width = raw_label.shape
    if returned_id != sample_id:
        raise AssertionError(f"Official Dataset returned unexpected sample ID: {returned_id}")
    if pre.shape != (3, height, width) or post.shape != (3, height, width):
        raise AssertionError(
            f"Official Dataset returned unexpected shapes: pre={pre.shape}, post={post.shape}"
        )
    if pre.dtype != np.float32 or post.dtype != np.float32:
        raise AssertionError(
            f"Official normalization should return float32: pre={pre.dtype}, post={post.dtype}"
        )
    expected_pre = np.transpose(
        (raw_pre[:, :, :3].astype(np.float32) - NORMALIZE_MEAN) / NORMALIZE_STD,
        (2, 0, 1),
    )
    expected_post = np.stack(
        [
            (raw_post.astype(np.float32) - NORMALIZE_MEAN[channel])
            / NORMALIZE_STD[channel]
            for channel in range(3)
        ],
        axis=0,
    )
    if not np.allclose(pre, expected_pre, rtol=0, atol=1e-6):
        raise AssertionError("Official Dataset produced unexpected optical normalization")
    if not np.allclose(post, expected_post, rtol=0, atol=1e-6):
        raise AssertionError("Official Dataset produced unexpected replicated-SAR normalization")
    if not np.array_equal(clf_label, raw_label.astype(np.float32)):
        raise AssertionError("Official Dataset changed class labels in test-mode loading")
    expected_loc = raw_label.astype(np.float32).copy()
    expected_loc[expected_loc > 0] = 1
    if not np.array_equal(loc_label, expected_loc):
        raise AssertionError("Official Dataset produced an unexpected localization label")


def center_crop(array: np.ndarray, crop_size: int) -> tuple[np.ndarray, dict[str, int]]:
    height, width = array.shape[-2:]
    if crop_size == 0:
        crop_size = min(height, width)
        if height != width:
            raise ValueError(
                "--forward-size 0 requires square input for this smoke test; "
                f"found {height}x{width}"
            )
    if crop_size < 16 or crop_size > min(height, width):
        raise ValueError(
            f"--forward-size must be between 16 and {min(height, width)}, got {crop_size}"
        )
    if crop_size % 16:
        raise ValueError(
            "--forward-size must be divisible by 16 because the official UNet pools four times"
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    cropped = array[..., top : top + crop_size, left : left + crop_size]
    return cropped, {"top": top, "left": left, "height": crop_size, "width": crop_size}


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch wheel and verify the NVIDIA driver."
            )
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:  # PyTorch versions older than the weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a state dict or a supported checkpoint mapping")
    model.load_state_dict(checkpoint, strict=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    train_list = args.train_list.expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {dataset_root}")

    train_ids = read_train_list(train_list)
    sample_id, event, paths = select_sample(dataset_root, train_ids, args.sample_id)

    raw_pre = np.asarray(imageio.imread(paths["pre_event"]))
    raw_post = np.asarray(imageio.imread(paths["post_event"]))
    raw_label = np.asarray(imageio.imread(paths["target"]))
    validate_raw_arrays(raw_pre, raw_post, raw_label, args.expected_size)

    # type="test" disables random crop/flip/rotation while preserving the exact
    # official read, channel selection, SAR replication, and normalization path.
    dataset = MultimodalDamageAssessmentDatset(
        str(dataset_root), [sample_id], crop_size=640, max_iters=None, type="test"
    )
    pre, post, loc_label, clf_label, returned_id = dataset[0]
    validate_official_loader_result(
        pre,
        post,
        loc_label,
        clf_label,
        raw_pre,
        raw_post,
        raw_label,
        sample_id,
        returned_id,
    )

    combined = np.concatenate((pre, post), axis=0)
    forward_array, crop = center_crop(combined, args.forward_size)
    input_tensor = torch.from_numpy(np.ascontiguousarray(forward_array)).unsqueeze(0)

    model = UNet(in_channels=6, num_classes=4)
    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        load_checkpoint(model, checkpoint_path)

    device = resolve_device(args.device)
    model.eval().to(device)
    input_tensor = input_tensor.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        with torch.inference_mode():
            logits = model(input_tensor)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA ran out of memory during the single forward pass. Close other GPU "
            "programs, keep --forward-size 640 and try --device cpu, or use a smaller "
            "multiple of 16 only as a hardware smoke-test variant."
        ) from exc

    expected_output_shape = (1, 4, crop["height"], crop["width"])
    if tuple(logits.shape) != expected_output_shape:
        raise AssertionError(
            f"Unexpected UNet output shape: {tuple(logits.shape)}; "
            f"expected {expected_output_shape}"
        )
    if not torch.isfinite(logits).all():
        raise AssertionError("UNet output contains NaN or infinite values")
    prediction_values = torch.unique(logits.argmax(dim=1)).cpu().tolist()

    device_name = platform.processor() or "CPU"
    peak_memory_mib = None
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        peak_memory_mib = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)

    return {
        "status": "passed",
        "scope": (
            "one dataset-root TIFF triplet and one inference-only forward pass; "
            "no backward pass, parameter update, or metric claim"
        ),
        "sample": {
            "id": sample_id,
            "event": event,
            "source_split": "train",
            "train_list": str(train_list),
            "paper_nine_event_member": True,
            "files": {name: str(path) for name, path in paths.items()},
        },
        "raw_tiff": {
            "pre_event": {
                **array_summary(raw_pre),
                "bands": int(raw_pre.shape[2]),
                "official_channels_used": [0, 1, 2],
            },
            "post_event": {**array_summary(raw_post), "bands": 1},
            "target": array_summary(raw_label, include_unique=True),
        },
        "official_loader": {
            "mode": "test (augmentation disabled)",
            "pre_shape": list(pre.shape),
            "post_shape": list(post.shape),
            "localization_label_shape": list(loc_label.shape),
            "classification_label_shape": list(clf_label.shape),
            "input_dtype": str(pre.dtype),
            "sar_source_replicated_then_channelwise_normalized": True,
            "classification_label_values": [int(value) for value in np.unique(clf_label)],
            "localization_label_values": [int(value) for value in np.unique(loc_label)],
        },
        "forward": {
            "requested_device": args.device,
            "resolved_device": str(device),
            "device_name": device_name,
            "forward_crop": crop,
            "input_shape": list(input_tensor.shape),
            "input_dtype": str(input_tensor.dtype),
            "output_shape": list(logits.shape),
            "output_dtype": str(logits.dtype),
            "output_all_finite": True,
            "argmax_values_present": [int(value) for value in prediction_values],
            "checkpoint_loaded": checkpoint_path is not None,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "cuda_peak_memory_mib": peak_memory_mib,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "imageio": __import__("imageio").__version__,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
    }


def main() -> None:
    args = parse_args()
    report = run(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
