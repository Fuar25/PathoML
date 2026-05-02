"""Tests for distillation teacher adapters."""

from __future__ import annotations

import torch

from distillation.models.teacher import TeacherMLP


def test_teacher_mlp_forward_exposes_classifier_direction():
  model = TeacherMLP(input_dim=4, hidden_dim=3, dropout=0.0)
  x = torch.randn(2, 4)

  out = model(x)

  assert out['hidden'].shape == (2, 3)
  assert out['logit'].shape == (2, 1)
  torch.testing.assert_close(
    out['class_weight'],
    model.net[3].weight.squeeze(0),
  )
