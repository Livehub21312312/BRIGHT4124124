#!/usr/bin/env python3
"""Audit a BRIGHT dataset without modifying any images, labels, or split files.

The checker mirrors the filename and array-shape assumptions in
``bda_benchmark/dataset/make_data_loader.py``:

* ``pre-event/<id>_pre_disaster.tif`` must be H x W x C with C >= 3;
* ``post-event/<id>_post_disaster.tif`` must be a two-dimensional SAR image;
* ``target/<id>_building_damage.tif`` must be a two-dimensional label image
  whose values are a subset of {0, 1, 2, 3}.

It writes a detailed JSON report and an Excel-friendly UTF-8 CSV report.  All
access to the dataset and the official split files is read-only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = REPO_ROOT / "bda_benchmark" / "dataset" / "splitname" / "standard_ML"

MODALITIES = {
    "pre_event": {
        "directory": "pre-event",
        "suffix": "_pre_disaster.tif",
    },
    "post_event": {
        "directory": "post-event",
        "suffix": "_post_disaster.tif",
    },
    "target": {
        "directory": "target",
        "suffix": "_building_damage.tif",
    },
}

SPLIT_FILES = {
    "train": "train_set.txt",
    "val": "val_set.txt",
    "test": "test_set.txt",
}

# Counts of non-empty entries at the pinned official repository commit.
EXPECTED_STANDARD_COUNTS = {"train": 2918, "val": 450, "test": 878}
EXPECTED_PAPER9_COUNTS = {"train": 2229, "val": 349, "test": 661}

PAPER9_EVENTS = (
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

ALLOWED_LABEL_VALUES = {0, 1, 2, 3}
SAMPLE_ID_RE = re.compile(r"^(?P<event>.+)_(?P<tile>\d{8})$")

CSV_FIELDS = (
    "record_type",
    "sample_id",
    "event",
    "splits",
    "in_train",
    "in_val",
    "in_test",
    "in_paper9",
    "complete_triplet",
    "dimensions_match",
    "pre_event_path",
    "pre_event_read_ok",
    "pre_event_shape",
    "pre_event_bands",
    "pre_event_dtype",
    "pre_event_min",
    "pre_event_max",
    "post_event_path",
    "post_event_read_ok",
    "post_event_shape",
    "post_event_bands",
    "post_event_dtype",
    "post_event_min",
    "post_event_max",
    "target_path",
    "target_read_ok",
    "target_shape",
    "target_bands",
    "target_dtype",
    "target_min",
    "target_max",
    "target_label_values",
    "target_label_value_count",
    "error_codes",
    "warning_codes",
    "source_path",
    "notes",
)


def issue(
    severity: str,
    code: str,
    message: str,
    *,
    modality: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if modality is not None:
        result["modality"] = modality
    if path is not None:
        result["path"] = path
    return result


def sample_event(sample_id: str) -> str | None:
    match = SAMPLE_ID_RE.fullmatch(sample_id)
    return match.group("event") if match else None


def json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, complex):
        return str(value)
    return value


def read_split_file(path: Path, split_name: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "name": split_name,
            "path": str(path.resolve()),
            "exists": False,
            "raw_line_count": 0,
            "blank_line_count": 0,
            "entry_count": 0,
            "unique_count": 0,
            "entries": [],
            "duplicates": [],
            "invalid_sample_ids": [],
            "issues": [issue("error", "missing_split_file", f"Split file does not exist: {path}")],
        }

    text = path.read_text(encoding="utf-8-sig")
    raw_lines = text.splitlines()
    entries = [line.strip() for line in raw_lines if line.strip()]
    counts = Counter(entry.casefold() for entry in entries)
    canonical_to_original: dict[str, str] = {}
    for entry in entries:
        canonical_to_original.setdefault(entry.casefold(), entry)
    duplicates = [canonical_to_original[key] for key, count in counts.items() if count > 1]
    invalid = [entry for entry in entries if sample_event(entry) is None]
    issues = []
    if duplicates:
        issues.append(
            issue(
                "error",
                "duplicate_entries_within_split",
                f"{split_name} contains {len(duplicates)} duplicated sample id(s).",
            )
        )
    if invalid:
        issues.append(
            issue(
                "error",
                "invalid_sample_ids_in_split",
                f"{split_name} contains {len(invalid)} id(s) that do not match <event>_<8 digits>.",
            )
        )
    return {
        "name": split_name,
        "path": str(path.resolve()),
        "exists": True,
        "raw_line_count": len(raw_lines),
        "blank_line_count": sum(not line.strip() for line in raw_lines),
        "entry_count": len(entries),
        "unique_count": len(counts),
        "entries": entries,
        "duplicates": sorted(duplicates),
        "invalid_sample_ids": sorted(invalid),
        "issues": issues,
    }


def scan_modality(dataset_root: Path, modality: str) -> dict[str, Any]:
    config = MODALITIES[modality]
    directory = dataset_root / config["directory"]
    expected_suffix = config["suffix"]
    expected_suffix_folded = expected_suffix.casefold()
    files_by_id: dict[str, list[Path]] = defaultdict(list)
    display_ids: dict[str, str] = {}
    unexpected_tiffs: list[str] = []
    non_tiff_file_count = 0
    nested_tiffs: list[str] = []
    issues = []

    if not directory.is_dir():
        issues.append(
            issue(
                "error",
                "missing_modality_directory",
                f"Required directory does not exist: {directory}",
                modality=modality,
                path=str(directory.resolve()),
            )
        )
        return {
            "directory": str(directory.resolve()),
            "exists": False,
            "files_by_id": {},
            "display_ids": {},
            "matched_file_count": 0,
            "unique_sample_count": 0,
            "duplicate_sample_ids": {},
            "unexpected_tiffs": [],
            "nested_tiffs": [],
            "non_tiff_file_count": 0,
            "issues": issues,
        }

    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            continue
        filename_folded = path.name.casefold()
        if filename_folded.endswith(expected_suffix_folded):
            sample_id = path.name[: -len(expected_suffix)]
            key = sample_id.casefold()
            files_by_id[key].append(path)
            display_ids.setdefault(key, sample_id)
        elif path.suffix.casefold() in {".tif", ".tiff"}:
            unexpected_tiffs.append(str(path.resolve()))
        else:
            non_tiff_file_count += 1

    # The official loader only checks files directly inside each modality
    # directory.  Nested TIFFs are reported, never silently treated as matches.
    for path in directory.rglob("*"):
        if path.is_file() and path.parent != directory and path.suffix.casefold() in {".tif", ".tiff"}:
            nested_tiffs.append(str(path.resolve()))

    duplicates = {
        display_ids[key]: [str(path.resolve()) for path in paths]
        for key, paths in files_by_id.items()
        if len(paths) > 1
    }
    if duplicates:
        issues.append(
            issue(
                "error",
                "duplicate_modality_files",
                f"{modality} has {len(duplicates)} sample id(s) with multiple matching files.",
                modality=modality,
            )
        )
    if unexpected_tiffs:
        issues.append(
            issue(
                "warning",
                "unexpected_tiff_names",
                f"{modality} has {len(unexpected_tiffs)} TIFF file(s) outside the official naming rule.",
                modality=modality,
            )
        )
    if nested_tiffs:
        issues.append(
            issue(
                "warning",
                "nested_tiffs_not_used_by_official_loader",
                f"{modality} has {len(nested_tiffs)} nested TIFF file(s).",
                modality=modality,
            )
        )

    return {
        "directory": str(directory.resolve()),
        "exists": True,
        "files_by_id": dict(files_by_id),
        "display_ids": display_ids,
        "matched_file_count": sum(len(paths) for paths in files_by_id.values()),
        "unique_sample_count": len(files_by_id),
        "duplicate_sample_ids": duplicates,
        "unexpected_tiffs": unexpected_tiffs,
        "nested_tiffs": sorted(nested_tiffs),
        "non_tiff_file_count": non_tiff_file_count,
        "issues": issues,
    }


def inspect_image(path: Path, modality: str, expected_size: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": True,
        "read_ok": False,
        "size_bytes": path.stat().st_size,
        "shape": None,
        "ndim": None,
        "height": None,
        "width": None,
        "bands": None,
        "dtype": None,
        "min": None,
        "max": None,
        "nan_count": None,
        "label_values": None,
        "label_value_count": None,
        "label_values_truncated": False,
    }
    issues: list[dict[str, Any]] = []
    try:
        array = np.asarray(imageio.imread(path))
    except Exception as exc:  # image backends raise several exception types
        issues.append(
            issue(
                "error",
                "tiff_read_failed",
                f"Could not read {modality} TIFF: {type(exc).__name__}: {exc}",
                modality=modality,
                path=str(path.resolve()),
            )
        )
        return metadata, issues

    metadata["read_ok"] = True
    metadata["shape"] = [int(value) for value in array.shape]
    metadata["ndim"] = int(array.ndim)
    metadata["dtype"] = str(array.dtype)

    if array.size == 0:
        issues.append(
            issue("error", "empty_tiff", "TIFF contains no pixels.", modality=modality, path=str(path.resolve()))
        )
        return metadata, issues

    if np.issubdtype(array.dtype, np.complexfloating):
        issues.append(
            issue(
                "error",
                "complex_dtype_not_supported",
                f"{modality} has unsupported complex dtype {array.dtype}.",
                modality=modality,
                path=str(path.resolve()),
            )
        )
    elif np.issubdtype(array.dtype, np.number):
        if np.issubdtype(array.dtype, np.floating):
            metadata["nan_count"] = int(np.isnan(array).sum())
            if metadata["nan_count"]:
                issues.append(
                    issue(
                        "error",
                        "non_finite_pixels",
                        f"{modality} contains {metadata['nan_count']} NaN pixel(s).",
                        modality=modality,
                        path=str(path.resolve()),
                    )
                )
        finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array
        if finite.size:
            metadata["min"] = json_scalar(finite.min())
            metadata["max"] = json_scalar(finite.max())
    else:
        issues.append(
            issue(
                "error",
                "non_numeric_dtype",
                f"{modality} has non-numeric dtype {array.dtype}.",
                modality=modality,
                path=str(path.resolve()),
            )
        )

    if modality == "pre_event":
        if array.ndim != 3:
            issues.append(
                issue(
                    "error",
                    "invalid_pre_event_shape",
                    f"Official loader requires HxWxC pre-event data; found shape {array.shape}.",
                    modality=modality,
                    path=str(path.resolve()),
                )
            )
        else:
            metadata["height"], metadata["width"], metadata["bands"] = map(int, array.shape)
            if metadata["bands"] < 3:
                issues.append(
                    issue(
                        "error",
                        "insufficient_pre_event_bands",
                        f"Pre-event image has {metadata['bands']} band(s); at least 3 are required.",
                        modality=modality,
                        path=str(path.resolve()),
                    )
                )
            elif metadata["bands"] > 3:
                issues.append(
                    issue(
                        "warning",
                        "extra_pre_event_bands",
                        f"Pre-event image has {metadata['bands']} bands; official code uses only the first 3.",
                        modality=modality,
                        path=str(path.resolve()),
                    )
                )
    elif modality in {"post_event", "target"}:
        if array.ndim != 2:
            issues.append(
                issue(
                    "error",
                    f"invalid_{modality}_shape",
                    f"Official loader requires a two-dimensional {modality} array; found shape {array.shape}.",
                    modality=modality,
                    path=str(path.resolve()),
                )
            )
        else:
            metadata["height"], metadata["width"] = map(int, array.shape)
            metadata["bands"] = 1

    if metadata["height"] is not None and expected_size > 0:
        if metadata["height"] != expected_size or metadata["width"] != expected_size:
            issues.append(
                issue(
                    "error",
                    "unexpected_spatial_size",
                    f"Expected {expected_size}x{expected_size}; found {metadata['height']}x{metadata['width']}.",
                    modality=modality,
                    path=str(path.resolve()),
                )
            )

    if modality in {"pre_event", "post_event"} and array.dtype != np.uint8:
        issues.append(
            issue(
                "warning",
                "non_uint8_image_dtype",
                f"{modality} dtype is {array.dtype}; official normalization assumes 0-255 imagery.",
                modality=modality,
                path=str(path.resolve()),
            )
        )
    if modality in {"pre_event", "post_event"} and metadata["min"] is not None:
        if metadata["min"] < 0 or metadata["max"] > 255:
            issues.append(
                issue(
                    "warning",
                    "image_values_outside_0_255",
                    f"{modality} value range is [{metadata['min']}, {metadata['max']}].",
                    modality=modality,
                    path=str(path.resolve()),
                )
            )

    if modality == "target" and array.ndim == 2:
        values = np.unique(array)
        metadata["label_value_count"] = int(values.size)
        shown_values = values[:100]
        metadata["label_values"] = [json_scalar(value) for value in shown_values]
        metadata["label_values_truncated"] = values.size > shown_values.size
        invalid_values = []
        for value in values:
            scalar = json_scalar(value)
            if not isinstance(scalar, (int, float)) or not np.isfinite(scalar) or scalar not in ALLOWED_LABEL_VALUES:
                invalid_values.append(scalar)
        if invalid_values:
            preview = invalid_values[:20]
            issues.append(
                issue(
                    "error",
                    "invalid_label_values",
                    f"Target contains values outside {{0,1,2,3}}: {preview}",
                    modality=modality,
                    path=str(path.resolve()),
                )
            )
        if not np.issubdtype(array.dtype, np.integer):
            issues.append(
                issue(
                    "warning",
                    "non_integer_label_dtype",
                    f"Target dtype is {array.dtype}; integer TIFF labels are expected.",
                    modality=modality,
                    path=str(path.resolve()),
                )
            )

    return metadata, issues


def inspect_sample(
    key: str,
    display_id: str,
    scans: dict[str, dict[str, Any]],
    split_sets: dict[str, set[str]],
    expected_size: int,
) -> dict[str, Any]:
    event = sample_event(display_id)
    splits = [name for name in SPLIT_FILES if key in split_sets[name]]
    record: dict[str, Any] = {
        "sample_id": display_id,
        "canonical_sample_id": key,
        "event": event,
        "splits": splits,
        "in_paper9": event in PAPER9_EVENTS if event is not None else False,
        "complete_triplet": False,
        "dimensions_match": None,
        "modalities": {},
        "issues": [],
    }
    if event is None:
        record["issues"].append(
            issue("error", "invalid_sample_id", "Sample id does not match <event>_<8 digits>.")
        )

    for modality, scan in scans.items():
        paths = scan["files_by_id"].get(key, [])
        if not paths:
            record["modalities"][modality] = {
                "path": None,
                "exists": False,
                "read_ok": False,
            }
            record["issues"].append(
                issue(
                    "error",
                    f"missing_{modality}",
                    f"No file matching the official {modality} naming rule.",
                    modality=modality,
                )
            )
            continue
        if len(paths) > 1:
            record["issues"].append(
                issue(
                    "error",
                    f"duplicate_{modality}_files",
                    f"Found {len(paths)} matching files for this sample.",
                    modality=modality,
                )
            )
        selected = paths[0]
        expected_suffix = MODALITIES[modality]["suffix"]
        expected_name = display_id + expected_suffix
        if selected.name != expected_name:
            record["issues"].append(
                issue(
                    "warning",
                    "filename_case_or_spelling_difference",
                    f"Found '{selected.name}', expected exact official spelling '{expected_name}'.",
                    modality=modality,
                    path=str(selected.resolve()),
                )
            )
        metadata, image_issues = inspect_image(selected, modality, expected_size)
        record["modalities"][modality] = metadata
        record["issues"].extend(image_issues)

    record["complete_triplet"] = all(
        record["modalities"].get(modality, {}).get("exists", False) for modality in MODALITIES
    )
    dimensions = [
        (
            record["modalities"].get(modality, {}).get("height"),
            record["modalities"].get(modality, {}).get("width"),
        )
        for modality in MODALITIES
        if record["modalities"].get(modality, {}).get("height") is not None
    ]
    if len(dimensions) == len(MODALITIES):
        record["dimensions_match"] = len(set(dimensions)) == 1
        if not record["dimensions_match"]:
            record["issues"].append(
                issue(
                    "error",
                    "spatial_dimension_mismatch",
                    f"Paired TIFF dimensions differ: {dict(zip(MODALITIES, dimensions))}",
                )
            )
    return record


def issue_codes(record: dict[str, Any], severity: str) -> list[str]:
    return sorted({item["code"] for item in record["issues"] if item["severity"] == severity})


def modality_csv(record: dict[str, Any], modality: str, field: str) -> str:
    value = record["modalities"].get(modality, {}).get(field)
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def sample_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "record_type": "sample",
            "sample_id": record["sample_id"],
            "event": record["event"] or "",
            "splits": ";".join(record["splits"]),
            "in_train": "train" in record["splits"],
            "in_val": "val" in record["splits"],
            "in_test": "test" in record["splits"],
            "in_paper9": record["in_paper9"],
            "complete_triplet": record["complete_triplet"],
            "dimensions_match": record["dimensions_match"],
            "error_codes": ";".join(issue_codes(record, "error")),
            "warning_codes": ";".join(issue_codes(record, "warning")),
            "notes": " | ".join(item["message"] for item in record["issues"]),
        }
    )
    for modality in MODALITIES:
        for field in ("path", "read_ok", "shape", "bands", "dtype", "min", "max"):
            row[f"{modality}_{field}"] = modality_csv(record, modality, field)
    row["target_label_values"] = modality_csv(record, "target", "label_values")
    row["target_label_value_count"] = modality_csv(record, "target", "label_value_count")
    return row


def unmatched_csv_row(path: str, record_type: str, code: str) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "record_type": record_type,
            "source_path": path,
            "warning_codes": code,
            "notes": "File is not addressable by the official direct-path loader.",
        }
    )
    return row


def aggregate_event_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "discovered_or_referenced_samples": 0,
            "complete_triplets": 0,
            "samples_with_errors": 0,
            "split_counts": {name: 0 for name in SPLIT_FILES},
            "missing_by_modality": {name: 0 for name in MODALITIES},
            "paper9_event": False,
        }
    )
    for record in records:
        event = record["event"] or "<invalid-sample-id>"
        event_row = summary[event]
        event_row["discovered_or_referenced_samples"] += 1
        event_row["complete_triplets"] += int(record["complete_triplet"])
        event_row["samples_with_errors"] += int(bool(issue_codes(record, "error")))
        event_row["paper9_event"] = bool(record["in_paper9"])
        for split_name in record["splits"]:
            event_row["split_counts"][split_name] += 1
        for modality in MODALITIES:
            if not record["modalities"].get(modality, {}).get("exists", False):
                event_row["missing_by_modality"][modality] += 1
    return dict(sorted(summary.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a BRIGHT TIFF dataset against official naming and standard split files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=r"Dataset root containing pre-event, post-event, and target (Windows paths are supported).",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory containing train_set.txt, val_set.txt, and test_set.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for dataset_check_report.json and dataset_check_report.csv.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of concurrent TIFF readers (default: min(4, CPU count)).",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=1024,
        help="Expected square tile size; use 0 to disable this check (default: 1024).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress after this many samples; use 0 to disable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.expected_size < 0:
        raise SystemExit("--expected-size cannot be negative")

    dataset_root = args.dataset_root.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {dataset_root}")
    print(f"Split directory: {split_dir}")
    print("Scanning official split files and modality directories...")

    split_reports = {
        name: read_split_file(split_dir / filename, name) for name, filename in SPLIT_FILES.items()
    }
    split_sets = {
        name: {entry.casefold() for entry in report["entries"]}
        for name, report in split_reports.items()
    }
    scans = {modality: scan_modality(dataset_root, modality) for modality in MODALITIES}

    all_keys: set[str] = set().union(*split_sets.values())
    display_ids: dict[str, str] = {}
    for report in split_reports.values():
        for entry in report["entries"]:
            display_ids.setdefault(entry.casefold(), entry)
    for scan in scans.values():
        all_keys.update(scan["files_by_id"])
        for key, value in scan["display_ids"].items():
            display_ids.setdefault(key, value)

    ordered_keys = sorted(all_keys)
    print(f"Inspecting {len(ordered_keys)} discovered or split-referenced sample id(s)...")

    def inspect_key(key: str) -> dict[str, Any]:
        return inspect_sample(key, display_ids.get(key, key), scans, split_sets, args.expected_size)

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, record in enumerate(executor.map(inspect_key, ordered_keys), start=1):
            records.append(record)
            if args.progress_every and (index % args.progress_every == 0 or index == len(ordered_keys)):
                print(f"  inspected {index}/{len(ordered_keys)}")

    record_by_key = {record["canonical_sample_id"]: record for record in records}
    global_issues: list[dict[str, Any]] = []
    for report in split_reports.values():
        global_issues.extend(report["issues"])
    for scan in scans.values():
        global_issues.extend(scan["issues"])

    for split_name, report in split_reports.items():
        expected = EXPECTED_STANDARD_COUNTS[split_name]
        if report["entry_count"] != expected:
            global_issues.append(
                issue(
                    "error",
                    "unexpected_standard_split_count",
                    f"{split_name} has {report['entry_count']} non-empty entries; expected {expected}.",
                )
            )

    overlaps: dict[str, list[str]] = {}
    split_names = list(SPLIT_FILES)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            shared = sorted(split_sets[left] & split_sets[right])
            overlaps[f"{left}_vs_{right}"] = [display_ids.get(key, key) for key in shared]
            if shared:
                global_issues.append(
                    issue(
                        "error",
                        "cross_split_overlap",
                        f"{left} and {right} share {len(shared)} sample id(s).",
                    )
                )

    paper9_counts = {
        split_name: sum(sample_event(display_ids.get(key, key)) in PAPER9_EVENTS for key in keys)
        for split_name, keys in split_sets.items()
    }
    for split_name, expected in EXPECTED_PAPER9_COUNTS.items():
        if paper9_counts[split_name] != expected:
            global_issues.append(
                issue(
                    "error",
                    "unexpected_paper9_split_count",
                    f"{split_name} paper-9 subset has {paper9_counts[split_name]} samples; expected {expected}.",
                )
            )

    missing_split_samples = []
    missing_by_split_event: dict[str, dict[str, Any]] = {}
    for split_name, keys in split_sets.items():
        event_rows: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "referenced_samples": 0,
                "missing_any_modality": 0,
                "missing_by_modality": {modality: 0 for modality in MODALITIES},
            }
        )
        for key in sorted(keys):
            record = record_by_key[key]
            event = record["event"] or "<invalid-sample-id>"
            event_rows[event]["referenced_samples"] += 1
            missing_modalities = [
                modality
                for modality in MODALITIES
                if not record["modalities"].get(modality, {}).get("exists", False)
            ]
            if missing_modalities:
                event_rows[event]["missing_any_modality"] += 1
                for modality in missing_modalities:
                    event_rows[event]["missing_by_modality"][modality] += 1
                missing_split_samples.append(
                    {
                        "sample_id": record["sample_id"],
                        "event": record["event"],
                        "split": split_name,
                        "missing_modalities": missing_modalities,
                    }
                )
        missing_by_split_event[split_name] = dict(sorted(event_rows.items()))

    sample_error_count = sum(bool(issue_codes(record, "error")) for record in records)
    sample_warning_count = sum(bool(issue_codes(record, "warning")) for record in records)
    all_sample_issues = [item for record in records for item in record["issues"]]
    all_issues = global_issues + all_sample_issues
    error_counts = Counter(item["code"] for item in all_issues if item["severity"] == "error")
    warning_counts = Counter(item["code"] for item in all_issues if item["severity"] == "warning")

    referenced_keys = set().union(*split_sets.values())
    orphan_records = [record["sample_id"] for record in records if record["canonical_sample_id"] not in referenced_keys]
    standard_records = [record_by_key[key] for key in referenced_keys]
    paper9_keys = {
        key for key in referenced_keys if sample_event(display_ids.get(key, key)) in PAPER9_EVENTS
    }
    paper9_records = [record_by_key[key] for key in paper9_keys]
    complete_for_standard_splits = (
        not any(issue_codes(record, "error") for record in standard_records)
        and not any(overlaps.values())
        and all(split_reports[name]["entry_count"] == EXPECTED_STANDARD_COUNTS[name] for name in SPLIT_FILES)
    )
    paper9_ready = (
        not any(issue_codes(record, "error") for record in paper9_records)
        and not any(overlaps.values())
        and paper9_counts == EXPECTED_PAPER9_COUNTS
    )

    if error_counts:
        status = "failed"
        exit_code = 1
    elif warning_counts:
        status = "passed_with_warnings"
        exit_code = 0
    else:
        status = "passed"
        exit_code = 0

    public_scans = {}
    for modality, scan in scans.items():
        public_scans[modality] = {
            key: value
            for key, value in scan.items()
            if key not in {"files_by_id", "display_ids"}
        }

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "exit_code": exit_code,
        "dataset_root": str(dataset_root),
        "split_directory": str(split_dir),
        "configuration": {
            "workers": args.workers,
            "expected_size": args.expected_size,
            "allowed_label_values": sorted(ALLOWED_LABEL_VALUES),
            "official_naming": {
                modality: {
                    "directory": config["directory"],
                    "suffix": config["suffix"],
                }
                for modality, config in MODALITIES.items()
            },
            "paper9_events": list(PAPER9_EVENTS),
            "expected_standard_counts": EXPECTED_STANDARD_COUNTS,
            "expected_paper9_counts": EXPECTED_PAPER9_COUNTS,
        },
        "summary": {
            "discovered_or_referenced_sample_count": len(records),
            "standard_split_unique_sample_count": len(referenced_keys),
            "paper9_unique_sample_count": len(paper9_keys),
            "complete_triplet_count": sum(record["complete_triplet"] for record in records),
            "sample_count_with_errors": sample_error_count,
            "sample_count_with_warnings": sample_warning_count,
            "missing_split_reference_count": len(missing_split_samples),
            "orphan_sample_count": len(orphan_records),
            "paper9_counts": paper9_counts,
            "paper9_counts_match_expected": paper9_counts == EXPECTED_PAPER9_COUNTS,
            "cross_split_overlap_count": sum(len(values) for values in overlaps.values()),
            "complete_for_all_current_standard_splits": complete_for_standard_splits,
            "paper9_subset_ready": paper9_ready,
            "error_counts_by_code": dict(sorted(error_counts.items())),
            "warning_counts_by_code": dict(sorted(warning_counts.items())),
        },
        "splits": {
            name: {key: value for key, value in split_report.items() if key != "entries"}
            for name, split_report in split_reports.items()
        },
        "modality_scans": public_scans,
        "cross_split_overlaps": overlaps,
        "missing_split_samples": missing_split_samples,
        "missing_by_split_event": missing_by_split_event,
        "orphan_samples_not_in_standard_splits": sorted(orphan_records),
        "event_summary": aggregate_event_summary(records),
        "global_issues": global_issues,
        "samples": records,
    }

    json_path = output_dir / "dataset_check_report.json"
    csv_path = output_dir / "dataset_check_report.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_rows = [sample_csv_row(record) for record in records]
    for modality, scan in scans.items():
        csv_rows.extend(
            unmatched_csv_row(path, f"unexpected_{modality}_tiff", "unexpected_tiff_names")
            for path in scan["unexpected_tiffs"]
        )
        csv_rows.extend(
            unmatched_csv_row(path, f"nested_{modality}_tiff", "nested_tiffs_not_used_by_official_loader")
            for path in scan["nested_tiffs"]
        )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(csv_rows)

    print("Dataset check complete.")
    print(f"  status: {status}")
    print(f"  standard split entries: { {name: report['entry_count'] for name, report in split_reports.items()} }")
    print(f"  paper-9 entries: {paper9_counts} (expected {EXPECTED_PAPER9_COUNTS})")
    print(f"  cross-split overlaps: {sum(len(values) for values in overlaps.values())}")
    print(f"  missing split references: {len(missing_split_samples)}")
    print(f"  samples with errors: {sample_error_count}")
    print(f"  samples with warnings: {sample_warning_count}")
    print(f"  JSON report: {json_path}")
    print(f"  CSV report: {csv_path}")
    if exit_code:
        print("Integrity errors were found; reports were still written. Exit code: 1")
    return exit_code


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
