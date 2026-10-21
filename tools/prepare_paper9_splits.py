#!/usr/bin/env python3
"""Create derived, read-only-filtered manifests for the historical 9 events.

The official standard_ML files are only read.  Filtered copies and a provenance
manifest are written to a separate output directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "bda_benchmark"
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
DEFAULT_SPLIT_ROOT = BASELINE_ROOT / "dataset" / "splitname" / "standard_ML"
EXPECTED_COUNTS = {"train": 2229, "val": 349, "test": 661}


def event_for_sample(sample_id: str) -> str | None:
    for event in PAPER_EVENTS:
        if sample_id.startswith(f"{event}_"):
            return event
    return None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_unique_ids(path: Path) -> tuple[list[str], str]:
    payload = path.read_bytes()
    ids = [line.strip() for line in payload.decode("utf-8").splitlines()]
    ids = [sample_id for sample_id in ids if sample_id]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate IDs in {path}: {sorted(duplicates)[:10]}")
    return ids, sha256_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter current official standard_ML lists to nine events."
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper9_splits",
    )
    args = parser.parse_args()

    selected: dict[str, list[str]] = {}
    sources: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        source = args.split_root / f"{split}_set.txt"
        if not source.is_file():
            raise FileNotFoundError(f"Official split file not found: {source}")
        all_ids, source_sha256 = read_unique_ids(source)
        ids = [sample_id for sample_id in all_ids if event_for_sample(sample_id)]
        if len(ids) != EXPECTED_COUNTS[split]:
            raise ValueError(
                f"Expected {EXPECTED_COUNTS[split]} {split} IDs, got {len(ids)}"
            )
        selected[split] = ids
        sources[split] = {
            "path": str(source.resolve()),
            "all_event_count": len(all_ids),
            "sha256": source_sha256,
        }

    overlaps: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        duplicate_ids = sorted(set(selected[left]) & set(selected[right]))
        overlaps[f"{left}_{right}"] = duplicate_ids
        if duplicate_ids:
            raise ValueError(f"{left}/{right} overlap: {duplicate_ids[:10]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}
    for split, ids in selected.items():
        payload = "".join(f"{sample_id}\n" for sample_id in ids).encode("utf-8")
        output = args.output_dir / f"{split}_set_paper9.txt"
        output.write_bytes(payload)
        outputs[split] = {
            "path": str(output.resolve()),
            "count": len(ids),
            "sha256": sha256_bytes(payload),
            "event_counts": dict(
                sorted(Counter(event_for_sample(sample_id) for sample_id in ids).items())
            ),
        }

    manifest = {
        "status": "passed",
        "scope_note": (
            "These are derived manifests for the historical nine-event subset, "
            "not the ESSD 2025 final 14-event standard split."
        ),
        "events": list(PAPER_EVENTS),
        "official_sources_unchanged": True,
        "sources": sources,
        "outputs": outputs,
        "overlap_counts": {name: len(ids) for name, ids in overlaps.items()},
    }
    manifest_path = args.output_dir / "paper9_split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
