#!/usr/bin/env python3
"""Forward-pass smoke tests for both archived ARViT source variants."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DISTANCE_SOURCE = ROOT / "legacy_sources" / "distance_based"
REGION_SOURCE = ROOT / "legacy_sources" / "region_similarity_based"

sys.path.insert(0, str(DISTANCE_SOURCE))
from ARViT2D.ARViT2D import ARViT2D  # noqa: E402

sys.path.insert(0, str(REGION_SOURCE))
from ARViT.ARViT import ARViT  # noqa: E402


def check_outputs(name: str, outputs: list, batch_size: int, classes: int) -> None:
    logits, reduced_attention, raw_attention, auxiliary = outputs
    assert logits.shape == (batch_size, classes), (name, logits.shape)
    assert len(reduced_attention) == 1
    assert len(raw_attention) == 1
    assert reduced_attention[0].shape == (batch_size, 256, 256)
    assert raw_attention[0].shape == (batch_size, 256, 256)
    assert auxiliary.shape == (batch_size, 256, 256)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(reduced_attention[0]).all()
    assert torch.isfinite(auxiliary).all()


def main() -> None:
    torch.manual_seed(1234)
    torch.set_num_threads(2)
    batch_size = 1
    classes = 3
    images = torch.rand(batch_size, 3, 256, 256)

    distance_model = ARViT2D(
        num_encoder_layers=1,
        nhead=4,
        num_classes=classes,
        batch_size=batch_size,
        hidden_dim=32,
        image_h=256,
        image_w=256,
        grid_l=16,
    ).eval()

    region_model = ARViT(
        num_encoder_layers=1,
        nhead=4,
        num_classes=classes,
        batch_size=batch_size,
        hidden_dim=32,
        image_h=256,
        image_w=256,
        grid_l=16,
        gm_patch=16,
    ).eval()

    with torch.no_grad():
        distance_outputs = distance_model(images)
        region_outputs = region_model(images)

    check_outputs("distance ARViT2D", distance_outputs, batch_size, classes)
    check_outputs("region ARViT", region_outputs, batch_size, classes)

    print("PASS: both archived ARViT variants instantiate, run on CPU, and expose attention maps")
    print(f"torch={torch.__version__}; attention_shape={tuple(distance_outputs[1][0].shape)}")


if __name__ == "__main__":
    main()
