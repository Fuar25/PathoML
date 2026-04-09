"""Tests for compositional distillation losses."""

from __future__ import annotations

import torch

from distillation.experiments.common import build_condition_name, describe_loss_design
from distillation.losses import (
  CompositeDistillationLoss,
  ContrastiveTeacherDiscriminationLoss,
  CosineAttentionLogitLoss,
  DiscriminationAttentionLogitLoss,
  HiddenLoss,
  RKDAngleLoss,
  RKDDistanceLoss,
  SoftLabelLoss,
  TaskLoss,
  WeightedTerm,
)
from distillation.losses.attention import RelationalTGALoss
from distillation.losses.relational import RKDLoss
from distillation.losses.standard import StandardKDLoss


def _toy_batch():
  torch.manual_seed(0)
  batch_size, n_instances, hidden_dim = 4, 3, 6
  s_out = {
    'logits': torch.randn(batch_size, 1),
    'hidden': torch.randn(batch_size, hidden_dim),
    'proj': torch.randn(batch_size, hidden_dim),
    'encoded': torch.randn(batch_size, n_instances, hidden_dim),
    'encoded_proj': torch.randn(batch_size, n_instances, hidden_dim),
    'attn_logits': torch.randn(batch_size, n_instances),
    'mask': torch.tensor(
      [[True, True, False], [True, True, True], [True, False, False], [True, True, True]]
    ),
  }
  t_out = {
    'logit': torch.randn(batch_size, 1),
    'hidden': torch.randn(batch_size, hidden_dim),
  }
  labels = torch.randint(0, 2, (batch_size,), dtype=torch.float32)
  return s_out, t_out, labels


def test_composite_loss_describe_and_slug():
  distill_loss = CompositeDistillationLoss([
    TaskLoss(),
    HiddenLoss(),
    SoftLabelLoss(temperature=4.0),
    WeightedTerm(RKDAngleLoss(), 2.0),
  ])

  assert describe_loss_design(distill_loss) == (
    "L_task + L_hidden + L_soft_label(T=4) + 2*L_rkd_angle"
  )
  assert build_condition_name('rkd', distill_loss) == (
    "rkd_task_hidden_soft_label_t4p0_rkd_angle_2p0"
  )


def test_standard_kd_wrapper_matches_composite_terms():
  s_out, t_out, labels = _toy_batch()
  wrapped = StandardKDLoss(alpha=1.0, beta=1.0, temperature=4.0)
  composite = CompositeDistillationLoss([
    TaskLoss(),
    HiddenLoss(),
    SoftLabelLoss(temperature=4.0),
  ])

  torch.testing.assert_close(
    wrapped(s_out, t_out, labels),
    composite(s_out, t_out, labels),
  )


def test_rkd_wrapper_matches_composite_terms():
  s_out, t_out, labels = _toy_batch()
  wrapped = RKDLoss(gamma_d=1.0, gamma_a=2.0)
  composite = CompositeDistillationLoss([
    TaskLoss(),
    RKDDistanceLoss(),
    WeightedTerm(RKDAngleLoss(), 2.0),
  ])

  torch.testing.assert_close(
    wrapped(s_out, t_out, labels),
    composite(s_out, t_out, labels),
  )


def test_rtga_wrapper_matches_composite_terms():
  s_out, t_out, labels = _toy_batch()
  wrapped = RelationalTGALoss(gamma=1.0, lam=0.1, tau=1.0)
  composite = CompositeDistillationLoss([
    TaskLoss(),
    DiscriminationAttentionLogitLoss(),
    WeightedTerm(ContrastiveTeacherDiscriminationLoss(tau=1.0), 0.1),
  ])

  torch.testing.assert_close(
    wrapped(s_out, t_out, labels),
    composite(s_out, t_out, labels),
  )


def test_tga_condition_name_uses_logit_space_term_slug():
  distill_loss = CompositeDistillationLoss([
    TaskLoss(),
    CosineAttentionLogitLoss(),
  ])

  assert build_condition_name('tga', distill_loss) == 'tga_task_attn_cosine_logits'
