#!/usr/bin/env python3
"""CPU smoke test for the unmodified BRIGHT set-level UNet baseline.

The test creates a temporary BRIGHT-style TIFF triplet, exercises the official
Dataset and UNet, performs one training forward/backward pass, performs an
evaluation-mode prediction, and checks the official Evaluator implementation.
It intentionally does not import train_UNet.py or infer_UNet.py because those
entry points unconditionally require CUDA.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F


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


def _write_sample(root: Path, sample_id: str, size: int) -> np.ndarray:
    for directory in ("pre-event", "post-event", "target"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    yy, xx = np.indices((size, size))
    pre = np.stack(
        (
            (xx * 7 + yy * 3) % 256,
            (xx * 5 + 31) % 256,
            (yy * 11 + 17) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    post = ((xx * 13 + yy * 9) % 256).astype(np.uint8)

    # Four equally represented classes: background, intact, damaged, destroyed.
    label = np.zeros((size, size), dtype=np.uint8)
    half = size // 2
    label[:half, half:] = 1
    label[half:, :half] = 2
    label[half:, half:] = 3

    imageio.imwrite(root / "pre-event" / f"{sample_id}_pre_disaster.tif", pre)
    imageio.imwrite(root / "post-event" / f"{sample_id}_post_disaster.tif", post)
    imageio.imwrite(root / "target" / f"{sample_id}_building_damage.tif", label)
    return label


def _check_metrics(label: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    evaluator = Evaluator(num_class=4)
    evaluator.add_batch(label[None, ...], prediction[None, ...])

    perfect = Evaluator(num_class=4)
    perfect.add_batch(label[None, ...], label[None, ...])
    perfect_iou = perfect.Intersection_over_Union()
    if not np.allclose(perfect_iou, np.ones(4), atol=1e-6):
        raise AssertionError(f"Perfect prediction did not yield unit IoU: {perfect_iou}")

    loc_label = label.copy()
    loc_label[loc_label > 0] = 1
    perfect_loc = Evaluator(num_class=2)
    perfect_loc.add_batch(loc_label[None, ...], loc_label[None, ...])

    perfect_damage = Evaluator(num_class=4)
    damage_mask = loc_label > 0
    perfect_damage.add_batch(label[damage_mask], label[damage_mask])
    damage_f1 = perfect_damage.Damage_F1_score()

    return {
        "random_weight_oa": float(evaluator.Pixel_Accuracy()),
        "random_weight_miou": float(evaluator.Mean_Intersection_over_Union()),
        "random_weight_iou": evaluator.Intersection_over_Union().tolist(),
        "perfect_localization_f1": float(perfect_loc.Pixel_F1_score()),
        "perfect_damage_f1": damage_f1.tolist(),
        "perfect_miou": float(perfect.Mean_Intersection_over_Union()),
    }


def main() -> None:
    size = 32
    sample_id = "synthetic-event_00000000"
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    with tempfile.TemporaryDirectory(prefix="bright-unet-smoke-") as temp_dir:
        data_root = Path(temp_dir)
        expected_label = _write_sample(data_root, sample_id, size)

        dataset = MultimodalDamageAssessmentDatset(
            str(data_root), [sample_id], crop_size=size, max_iters=None, type="test"
        )
        pre, post, loc_label, clf_label, returned_id = dataset[0]
        if returned_id != sample_id:
            raise AssertionError(f"Unexpected sample id: {returned_id}")
        if pre.shape != (3, size, size) or post.shape != (3, size, size):
            raise AssertionError(f"Unexpected input shapes: pre={pre.shape}, post={post.shape}")
        if not np.array_equal(clf_label, expected_label):
            raise AssertionError("The official Dataset changed the test-mode class labels.")
        if not np.array_equal(loc_label, (expected_label > 0).astype(expected_label.dtype)):
            raise AssertionError("The official Dataset produced an unexpected localization mask.")

        inputs = torch.from_numpy(np.concatenate((pre, post), axis=0)).unsqueeze(0)
        target = torch.from_numpy(clf_label).long().unsqueeze(0)
        model = UNet(in_channels=6, num_classes=4)

        model.train()
        logits = model(inputs)
        if logits.shape != (1, 4, size, size):
            raise AssertionError(f"Unexpected UNet output shape: {tuple(logits.shape)}")
        ce_loss = F.cross_entropy(logits, target)
        lovasz_loss = lovasz_softmax(F.softmax(logits, dim=1), target, ignore=255)
        official_code_loss = ce_loss + 0.75 * lovasz_loss
        if not torch.isfinite(official_code_loss):
            raise AssertionError(f"Non-finite training loss: {official_code_loss.item()}")
        official_code_loss.backward()
        if model.encoder1.block[0].weight.grad is None:
            raise AssertionError("Backward pass did not produce an encoder gradient.")

        model.eval()
        with torch.inference_mode():
            inference_logits = model(inputs)
            prediction = inference_logits.argmax(dim=1).squeeze(0).cpu().numpy()

        metrics = _check_metrics(expected_label, prediction)
        report = {
            "status": "ok",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "input_shape": list(inputs.shape),
            "output_shape": list(inference_logits.shape),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "cross_entropy": float(ce_loss.detach()),
            "lovasz_softmax": float(lovasz_loss.detach()),
            "code_loss_ce_plus_0_75_lovasz": float(official_code_loss.detach()),
            "metrics": metrics,
        }
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
