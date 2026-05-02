"""Tests for registered patch fusion MIL."""

from __future__ import annotations

import torch

from teacher.models.registered_patch_fusion import RegisteredPatchFusionMIL


def test_registered_patch_fusion_mil_forward_shapes():
  model = RegisteredPatchFusionMIL(
    input_dim=12,
    hidden_dim=8,
    num_classes=1,
    num_modalities=3,
    modality_hidden_dim=4,
    dropout=0.0,
    modality_dropout=0.0,
  )
  batch = {
    'features': torch.randn(2, 5, 12),
    'modality_mask': torch.tensor([
      [
        [True, True, True],
        [True, False, True],
        [False, True, False],
        [False, False, False],
        [False, False, False],
      ],
      [
        [True, True, True],
        [True, True, True],
        [True, True, False],
        [True, False, False],
        [False, True, True],
      ],
    ]),
    'mask': torch.tensor([
      [True, True, True, False, False],
      [True, True, True, True, True],
    ]),
  }

  output = model(batch)

  assert output['logits'].shape == (2, 1)
  assert output['bag_embeddings'].shape == (2, 8)
  assert output['attention'].shape == (2, 5)
  assert torch.allclose(output['attention'][0, 3:], torch.zeros(2))


def test_registered_patch_fusion_mil_rejects_bad_input_dim():
  try:
    RegisteredPatchFusionMIL(
      input_dim=10,
      hidden_dim=8,
      num_classes=1,
      num_modalities=3,
    )
  except ValueError as exc:
    assert "divisible" in str(exc)
  else:
    raise AssertionError("Expected ValueError")
