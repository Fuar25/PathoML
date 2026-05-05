"""Tests for registered patch fusion MIL."""

from __future__ import annotations

import torch

from teacher.models.registered_patch_fusion import (
  RegisteredPatchCoordFusionMIL,
  RegisteredPatchFusionMIL,
  RegisteredPatchPolyCoordFusionMIL,
  RegisteredPatchPolyCoordStainAffineGateFusionMIL,
  RegisteredPatchPolyCoordStainBiasCoordGateFusionMIL,
)


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


def test_registered_patch_coord_fusion_mil_forward_shapes_with_masks():
  model = RegisteredPatchCoordFusionMIL(
    input_dim=12,
    hidden_dim=8,
    num_classes=1,
    num_modalities=3,
    modality_hidden_dim=4,
    coord_hidden_dim=3,
    dropout=0.0,
    modality_dropout=0.0,
  )
  batch = {
    'features': torch.randn(2, 5, 12),
    'coords': torch.tensor([
      [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 0.0], [0.0, 0.0]],
      [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0], [5.5, 5.5]],
    ]),
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
  assert output['bag_embeddings'].shape == (2, 11)
  assert output['attention'].shape == (2, 5)
  assert torch.allclose(output['attention'][0, 3:], torch.zeros(2))
  assert torch.isfinite(output['logits']).all()
  assert torch.isfinite(output['bag_embeddings']).all()


def test_registered_patch_coord_fusion_mil_handles_constant_coords():
  model = RegisteredPatchCoordFusionMIL(
    input_dim=12,
    hidden_dim=8,
    num_classes=1,
    num_modalities=3,
    modality_hidden_dim=4,
    coord_hidden_dim=3,
    dropout=0.0,
    modality_dropout=0.0,
  )
  batch = {
    'features': torch.randn(2, 4, 12),
    'coords': torch.tensor([
      [[7.0, 11.0], [7.0, 11.0], [7.0, 11.0], [0.0, 0.0]],
      [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0]],
    ]),
    'mask': torch.tensor([
      [True, True, True, False],
      [True, True, True, True],
    ]),
  }

  output = model(batch)
  normalized = model._normalized_coords(batch['coords'], batch['mask'], torch.float32)

  assert output['logits'].shape == (2, 1)
  assert output['bag_embeddings'].shape == (2, 11)
  assert output['attention'].shape == (2, 4)
  assert torch.isfinite(output['logits']).all()
  assert torch.allclose(normalized, torch.zeros_like(normalized))


def test_registered_patch_polycoord_stain_affine_gate_forward_shapes_and_initial_equivalence():
  model_kwargs = {
    'input_dim': 12,
    'hidden_dim': 8,
    'num_classes': 1,
    'num_modalities': 3,
    'modality_hidden_dim': 4,
    'coord_hidden_dim': 3,
    'dropout': 0.0,
    'modality_dropout': 0.0,
    'attention_dim': 5,
  }
  torch.manual_seed(7)
  baseline = RegisteredPatchPolyCoordFusionMIL(**model_kwargs)
  torch.manual_seed(7)
  affine = RegisteredPatchPolyCoordStainAffineGateFusionMIL(**model_kwargs)
  baseline.eval()
  affine.eval()
  batch = {
    'features': torch.randn(2, 5, 12),
    'coords': torch.tensor([
      [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 0.0], [0.0, 0.0]],
      [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0], [5.5, 5.5]],
    ]),
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

  baseline_output = baseline(batch)
  affine_output = affine(batch)

  assert affine.stain_affine_scale.shape == (3,)
  assert affine.stain_affine_bias.shape == (3,)
  assert torch.allclose(affine.stain_affine_scale, torch.ones(3))
  assert torch.allclose(affine.stain_affine_bias, torch.zeros(3))
  assert affine_output['logits'].shape == (2, 1)
  assert affine_output['bag_embeddings'].shape == (2, 11)
  assert affine_output['attention'].shape == (2, 5)
  assert torch.allclose(affine_output['logits'], baseline_output['logits'])
  assert torch.allclose(affine_output['bag_embeddings'], baseline_output['bag_embeddings'])
  assert torch.allclose(affine_output['attention'], baseline_output['attention'])


def test_registered_patch_polycoord_stain_affine_gate_runs_after_mask():
  model = RegisteredPatchPolyCoordStainAffineGateFusionMIL(
    input_dim=12,
    hidden_dim=8,
    num_classes=1,
    num_modalities=3,
    modality_hidden_dim=4,
    coord_hidden_dim=3,
    dropout=0.0,
    modality_dropout=0.0,
  )
  with torch.no_grad():
    model.stain_affine_scale.copy_(torch.tensor([2.0, 3.0, 4.0]))
    model.stain_affine_bias.copy_(torch.tensor([0.5, -1.0, 1.5]))

  encoded = torch.ones(1, 2, 3, 4)
  modality_mask = torch.tensor([[
    [True, False, True],
    [False, True, False],
  ]])

  masked = model._apply_modality_mask(encoded, modality_mask)
  transformed = model._apply_encoded_modality_transform(masked)

  assert torch.allclose(transformed[0, 0, 0], torch.full((4,), 2.5))
  assert torch.allclose(transformed[0, 0, 1], torch.full((4,), -1.0))
  assert torch.allclose(transformed[0, 1, 0], torch.full((4,), 0.5))
  assert torch.allclose(transformed[0, 1, 1], torch.full((4,), 2.0))


def test_registered_patch_polycoord_stain_bias_coord_gate_forward_shapes_and_initial_equivalence():
  model_kwargs = {
    'input_dim': 12,
    'hidden_dim': 8,
    'num_classes': 1,
    'num_modalities': 3,
    'modality_hidden_dim': 4,
    'coord_hidden_dim': 3,
    'dropout': 0.0,
    'modality_dropout': 0.0,
    'attention_dim': 5,
  }
  torch.manual_seed(11)
  baseline = RegisteredPatchPolyCoordStainAffineGateFusionMIL(**model_kwargs)
  torch.manual_seed(11)
  gated = RegisteredPatchPolyCoordStainBiasCoordGateFusionMIL(**model_kwargs)
  baseline.eval()
  gated.eval()
  batch = {
    'features': torch.randn(2, 5, 12),
    'coords': torch.tensor([
      [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 0.0], [0.0, 0.0]],
      [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0], [5.5, 5.5]],
    ]),
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

  baseline_output = baseline(batch)
  gated_output = gated(batch)

  assert gated.stain_bias.shape == (3,)
  assert gated.coord_modality_gate.weight.shape == (3, 5)
  assert gated.coord_modality_gate.bias.shape == (3,)
  assert torch.allclose(gated.stain_bias, torch.zeros(3))
  assert torch.allclose(gated.coord_modality_gate.weight, torch.zeros(3, 5))
  assert torch.allclose(gated.coord_modality_gate.bias, torch.zeros(3))
  assert gated_output['logits'].shape == (2, 1)
  assert gated_output['bag_embeddings'].shape == (2, 11)
  assert gated_output['attention'].shape == (2, 5)
  assert torch.allclose(gated_output['logits'], baseline_output['logits'])
  assert torch.allclose(gated_output['bag_embeddings'], baseline_output['bag_embeddings'])
  assert torch.allclose(gated_output['attention'], baseline_output['attention'])


def test_registered_patch_polycoord_stain_bias_coord_gate_runs_after_mask():
  model = RegisteredPatchPolyCoordStainBiasCoordGateFusionMIL(
    input_dim=12,
    hidden_dim=8,
    num_classes=1,
    num_modalities=3,
    modality_hidden_dim=4,
    coord_hidden_dim=3,
    dropout=0.0,
    modality_dropout=0.0,
    coord_gate_scale=0.1,
  )
  with torch.no_grad():
    model.stain_bias.copy_(torch.tensor([0.5, -1.0, 1.5]))
    model.coord_modality_gate.weight.zero_()
    model.coord_modality_gate.bias.zero_()
    model.coord_modality_gate.weight[0, 0] = 2.0

  encoded = torch.ones(1, 2, 3, 4)
  modality_mask = torch.tensor([[
    [True, False, True],
    [False, True, False],
  ]])
  data_dict = {
    'coords': torch.tensor([[[0.0, 0.0], [10.0, 0.0]]]),
    'mask': torch.tensor([[True, True]]),
  }

  masked = model._apply_modality_mask(encoded, modality_mask)
  transformed = model._apply_encoded_modality_transform(masked, data_dict)
  normalized = model._normalized_coords(
    data_dict['coords'],
    data_dict['mask'],
    torch.float32,
  )
  gate_input = model._coord_encoder_input(normalized)
  he_gate = 1.0 + 0.1 * (
    2.0 * torch.sigmoid(2.0 * gate_input[..., 0]) - 1.0
  )

  assert torch.allclose(transformed[0, 0, 0], torch.full((4,), he_gate[0, 0] + 0.5))
  assert torch.allclose(transformed[0, 0, 1], torch.full((4,), -1.0))
  assert torch.allclose(transformed[0, 1, 0], torch.full((4,), 0.5))
  assert torch.allclose(transformed[0, 1, 1], torch.full((4,), 0.0))
