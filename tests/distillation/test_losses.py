"""Tests for compositional distillation losses."""

from __future__ import annotations

import pytest
import torch

from distillation.experiments.common import build_condition_name, describe_loss_design
from distillation.losses import (
  BatchContrastiveAttentionLoss,
  ClassAwareAttentionRankMarginLoss,
  ClassAwareCosineAttentionLogitLoss,
  CompositeDistillationLoss,
  ConfidenceGatedCosineAttentionLogitLoss,
  CosineAttentionLogitLoss,
  CosineAttentionRankLoss,
  DecoupledKnowledgeDistillationLoss,
  HiddenLoss,
  RKDAngleLoss,
  RKDDistanceLoss,
  SimilarityPreservingLoss,
  SoftDistributionAttentionLoss,
  SoftLabelLoss,
  TaskLoss,
  TopKCosineAttentionLogitLoss,
  WeightedTerm,
)
from distillation.losses.attention import TeacherGuidedAttnLoss
from distillation.losses.relational import RKDLoss
from distillation.losses.standard import StandardKDLoss


def _toy_batch():
  torch.manual_seed(0)
  batch_size, n_instances, hidden_dim = 4, 3, 6
  s_out = {
    'logits': torch.randn(batch_size, 1),
    'hidden': torch.randn(batch_size, hidden_dim),
    'encoded': torch.randn(batch_size, n_instances, hidden_dim),
    'attn_logits': torch.randn(batch_size, n_instances),
    'mask': torch.tensor(
      [[True, True, False], [True, True, True], [True, False, False], [True, True, True]]
    ),
  }
  t_out = {
    'logit': torch.randn(batch_size, 1),
    'hidden': torch.randn(batch_size, hidden_dim),
    'class_weight': torch.randn(hidden_dim),
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


def test_teacher_guided_attention_wrapper_matches_composite_terms():
  s_out, t_out, labels = _toy_batch()
  wrapped = TeacherGuidedAttnLoss(alpha=1.0, beta=1.0, temperature=4.0, gamma=1.0)
  composite = CompositeDistillationLoss([
    TaskLoss(),
    HiddenLoss(),
    SoftLabelLoss(temperature=4.0),
    CosineAttentionLogitLoss(),
  ])

  torch.testing.assert_close(
    wrapped(s_out, t_out, labels),
    composite(s_out, t_out, labels),
  )


def test_teacher_guided_attention_wrapper_omits_zero_weight_terms():
  distill_loss = TeacherGuidedAttnLoss(alpha=0.0, beta=0.0, gamma=0.0)

  assert describe_loss_design(distill_loss) == "L_task"
  assert build_condition_name('tga', distill_loss) == "tga_task"
  assert repr(distill_loss) == (
    "TeacherGuidedAttnLoss(alpha=0.0, beta=0.0, temperature=4.0, "
    "gamma=0.0, delta=0.0, tau=1.0)"
  )


def test_teacher_guided_attention_wrapper_rejects_mean_bypass_delta():
  with pytest.raises(NotImplementedError, match="delta/mean-bypass"):
    TeacherGuidedAttnLoss(delta=0.1)



def test_tga_condition_name_uses_logit_space_term_slug():
  distill_loss = CompositeDistillationLoss([
    TaskLoss(),
    CosineAttentionLogitLoss(),
  ])

  assert build_condition_name('tga', distill_loss) == 'tga_task_attn_cosine_logits_no_detach'


def test_tga_condition_name_supports_detached_logit_term_slug():
  distill_loss = CompositeDistillationLoss([
    TaskLoss(),
    CosineAttentionLogitLoss(detach_target_encoded=True),
  ])

  assert build_condition_name('tga', distill_loss) == 'tga_task_attn_cosine_logits_detach'


def test_cosine_attention_logit_loss_describe_and_slug_variants():
  no_detach = CosineAttentionLogitLoss()
  detach = CosineAttentionLogitLoss(detach_target_encoded=True)

  assert no_detach.describe() == 'L_attn_cosine_no_detach'
  assert no_detach.slug() == 'attn_cosine_logits_no_detach'
  assert detach.describe() == 'L_attn_cosine_detach'
  assert detach.slug() == 'attn_cosine_logits_detach'


def test_class_aware_cosine_attention_loss_describe_and_slug():
  loss = ClassAwareCosineAttentionLogitLoss()
  detach = ClassAwareCosineAttentionLogitLoss(
    hidden_weight=0.25,
    class_weight=0.75,
    detach_target_encoded=True,
  )

  assert loss.describe() == 'L_attn_class_aware_cosine_no_detach(h=0.5, c=0.5)'
  assert loss.slug() == 'attn_class_aware_cosine_h0p5_c0p5_no_detach'
  assert detach.describe() == 'L_attn_class_aware_cosine_detach(h=0.25, c=0.75)'
  assert detach.slug() == 'attn_class_aware_cosine_h0p25_c0p75_detach'


def test_class_aware_cosine_attention_loss_uses_signed_class_direction():
  loss = ClassAwareCosineAttentionLogitLoss(hidden_weight=0.0, class_weight=1.0)
  labels = torch.tensor([1.0, 0.0])
  t_out = {
    'hidden': torch.zeros(2, 2),
    'class_weight': torch.tensor([1.0, 0.0]),
  }
  encoded = torch.tensor([
    [[1.0, 0.0], [-1.0, 0.0]],
    [[1.0, 0.0], [-1.0, 0.0]],
  ])
  aligned_s_out = {
    'encoded': encoded,
    'attn_logits': torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
  }
  reversed_s_out = {
    'encoded': encoded,
    'attn_logits': torch.tensor([[-1.0, 1.0], [1.0, -1.0]]),
  }

  assert loss(aligned_s_out, t_out, labels) < loss(reversed_s_out, t_out, labels)


def test_tga_condition_name_supports_class_aware_cosine_term_slug():
  distill_loss = CompositeDistillationLoss([
    TaskLoss(),
    ClassAwareCosineAttentionLogitLoss(),
  ])

  assert build_condition_name('teacher_guided_attention', distill_loss) == (
    'teacher_guided_attention_task_attn_class_aware_cosine_h0p5_c0p5_no_detach'
  )


def test_class_aware_rank_margin_loss_describe_and_slug():
  loss = ClassAwareAttentionRankMarginLoss()
  no_detach = ClassAwareAttentionRankMarginLoss(
    hidden_weight=0.25,
    class_weight=0.75,
    top_ratio=0.5,
    margin=0.25,
    detach_target_encoded=False,
  )

  assert loss.describe() == (
    'L_attn_class_aware_rank_margin_detach(h=0.5, c=0.5, r=0.25, m=1)'
  )
  assert loss.slug() == (
    'attn_class_aware_rank_margin_h0p5_c0p5_r0p25_m1p0_detach'
  )
  assert no_detach.describe() == (
    'L_attn_class_aware_rank_margin_no_detach(h=0.25, c=0.75, r=0.5, m=0.25)'
  )
  assert no_detach.slug() == (
    'attn_class_aware_rank_margin_h0p25_c0p75_r0p5_m0p25_no_detach'
  )


def test_class_aware_rank_margin_loss_uses_signed_top_bottom_order():
  loss = ClassAwareAttentionRankMarginLoss(
    hidden_weight=0.0,
    class_weight=1.0,
    top_ratio=0.5,
    margin=1.0,
  )
  labels = torch.tensor([1.0, 0.0])
  t_out = {
    'hidden': torch.zeros(2, 2),
    'class_weight': torch.tensor([1.0, 0.0]),
  }
  encoded = torch.tensor([
    [[1.0, 0.0], [-1.0, 0.0]],
    [[1.0, 0.0], [-1.0, 0.0]],
  ])
  aligned_s_out = {
    'encoded': encoded,
    'attn_logits': torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
  }
  reversed_s_out = {
    'encoded': encoded,
    'attn_logits': torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
  }

  torch.testing.assert_close(
    loss(aligned_s_out, t_out, labels),
    torch.zeros(()),
  )
  assert loss(reversed_s_out, t_out, labels) > loss(aligned_s_out, t_out, labels)


def test_class_aware_rank_margin_loss_skips_single_instance_bags():
  loss = ClassAwareAttentionRankMarginLoss()
  labels = torch.tensor([1.0])
  t_out = {
    'hidden': torch.zeros(1, 2),
    'class_weight': torch.tensor([1.0, 0.0]),
  }
  s_out = {
    'encoded': torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]]),
    'attn_logits': torch.tensor([[10.0, -10.0]]),
    'mask': torch.tensor([[True, False]]),
  }

  torch.testing.assert_close(
    loss(s_out, t_out, labels),
    torch.zeros(()),
  )


def test_tga_condition_name_supports_class_aware_rank_margin_term_slug():
  distill_loss = CompositeDistillationLoss([
    TaskLoss(),
    ClassAwareAttentionRankMarginLoss(),
  ])

  assert build_condition_name('teacher_guided_attention', distill_loss) == (
    'teacher_guided_attention_task_'
    'attn_class_aware_rank_margin_h0p5_c0p5_r0p25_m1p0_detach'
  )


def test_confidence_gated_cosine_attention_loss_describe_and_slug():
  loss = ConfidenceGatedCosineAttentionLogitLoss()
  normalized = ConfidenceGatedCosineAttentionLogitLoss(normalize_by_gate=True)
  thresholded = ConfidenceGatedCosineAttentionLogitLoss(
    detach_target_encoded=True,
    min_confidence=0.25,
  )

  assert loss.describe() == 'L_attn_cosine_confidence_gated_no_detach'
  assert loss.slug() == 'attn_cosine_confidence_gated_no_detach'
  assert normalized.describe() == 'L_attn_cosine_confidence_gated_normalized_no_detach'
  assert normalized.slug() == 'attn_cosine_confidence_gated_normalized_no_detach'
  assert thresholded.describe() == (
    'L_attn_cosine_confidence_gated_detach(min_confidence=0.25)'
  )
  assert thresholded.slug() == (
    'attn_cosine_confidence_gated_detach_min_confidence0p25'
  )


def test_confidence_gated_cosine_attention_loss_suppresses_uncertain_teacher():
  loss = ConfidenceGatedCosineAttentionLogitLoss()
  labels = torch.zeros(2)
  s_out = {
    'encoded': torch.tensor([
      [[1.0, 0.0], [0.0, 1.0]],
      [[1.0, 0.0], [0.0, 1.0]],
    ]),
    'attn_logits': torch.tensor([
      [10.0, -10.0],
      [10.0, -10.0],
    ]),
  }
  confident_t_out = {
    'hidden': torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    'logit': torch.tensor([[20.0], [-20.0]]),
  }
  uncertain_t_out = {
    'hidden': torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    'logit': torch.tensor([[0.0], [0.0]]),
  }

  torch.testing.assert_close(
    loss(s_out, uncertain_t_out, labels),
    torch.zeros(()),
  )
  assert loss(s_out, confident_t_out, labels) > loss(s_out, uncertain_t_out, labels)


def test_normalized_confidence_gated_cosine_attention_loss_preserves_scale():
  normalized = ConfidenceGatedCosineAttentionLogitLoss(normalize_by_gate=True)
  ungated = CosineAttentionLogitLoss()
  labels = torch.zeros(2)
  s_out = {
    'encoded': torch.tensor([
      [[1.0, 0.0], [0.0, 1.0]],
      [[1.0, 0.0], [0.0, 1.0]],
    ]),
    'attn_logits': torch.tensor([
      [10.0, -10.0],
      [10.0, -10.0],
    ]),
  }
  confident_t_out = {
    'hidden': torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    'logit': torch.tensor([[20.0], [-20.0]]),
  }

  torch.testing.assert_close(
    normalized(s_out, confident_t_out, labels),
    ungated(s_out, confident_t_out, labels),
  )


def test_tga_condition_name_supports_confidence_gated_cosine_term_slug():
  distill_loss = CompositeDistillationLoss([
    TaskLoss(),
    ConfidenceGatedCosineAttentionLogitLoss(),
  ])

  assert build_condition_name('teacher_guided_attention', distill_loss) == (
    'teacher_guided_attention_task_attn_cosine_confidence_gated_no_detach'
  )


def test_tga_condition_name_supports_normalized_confidence_gated_cosine_term_slug():
  distill_loss = CompositeDistillationLoss([
    TaskLoss(),
    ConfidenceGatedCosineAttentionLogitLoss(normalize_by_gate=True),
  ])

  assert build_condition_name('teacher_guided_attention', distill_loss) == (
    'teacher_guided_attention_task_attn_cosine_confidence_gated_normalized_no_detach'
  )


def test_cosine_attention_rank_loss_prefers_correct_ordering():
  loss = CosineAttentionRankLoss()
  labels = torch.zeros(1)
  t_out = {'hidden': torch.tensor([[1.0, 0.0]])}
  encoded_proj = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]])

  aligned_s_out = {
    'encoded': encoded_proj,
    'attn_logits': torch.tensor([[3.0, 1.0, -1.0]]),
  }
  reversed_s_out = {
    'encoded': encoded_proj,
    'attn_logits': torch.tensor([[-1.0, 1.0, 3.0]]),
  }

  assert loss(aligned_s_out, t_out, labels) < loss(reversed_s_out, t_out, labels)


def test_cosine_attention_rank_loss_respects_mask():
  loss = CosineAttentionRankLoss()
  labels = torch.zeros(1)
  t_out = {'hidden': torch.tensor([[1.0, 0.0]])}
  encoded_proj = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
  s_out = {
    'encoded': encoded_proj,
    'attn_logits': torch.tensor([[2.0, -1.0, 5.0]]),
    'mask': torch.tensor([[True, True, False]]),
  }
  masked_expected = torch.nn.functional.softplus(torch.tensor(-3.0))

  torch.testing.assert_close(loss(s_out, t_out, labels), masked_expected)


def test_tga_condition_name_supports_cosine_rank_term_slug():
  distill_loss = CompositeDistillationLoss([
    TaskLoss(),
    CosineAttentionRankLoss(),
  ])

  assert build_condition_name('tga', distill_loss) == 'tga_task_attn_cosine_rank'


def test_topk_cosine_attention_logit_loss_describe_and_slug():
  loss = TopKCosineAttentionLogitLoss(topk_ratio=0.25)

  assert loss.describe() == 'L_attn_cosine_topk(r=0.25)'
  assert loss.slug() == 'attn_cosine_topk_r0p25'


def test_topk_cosine_attention_logit_loss_uses_only_topk_patches():
  loss = TopKCosineAttentionLogitLoss(topk_ratio=0.5)
  labels = torch.zeros(1)
  t_out = {'hidden': torch.tensor([[1.0, 0.0]])}
  encoded_proj = torch.tensor([[[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]]])
  better_s_out = {
    'encoded': encoded_proj,
    'attn_logits': torch.tensor([[1.0, 0.8, 7.0, -9.0]]),
  }
  worse_s_out = {
    'encoded': encoded_proj,
    'attn_logits': torch.tensor([[-1.0, -0.8, 7.0, -9.0]]),
  }

  assert loss(better_s_out, t_out, labels) < loss(worse_s_out, t_out, labels)


def test_topk_cosine_attention_logit_loss_respects_valid_mask():
  loss = TopKCosineAttentionLogitLoss(topk_ratio=0.5)
  labels = torch.zeros(1)
  t_out = {'hidden': torch.tensor([[1.0, 0.0]])}
  encoded_proj = torch.tensor([[[1.0, 0.0], [0.7, 0.3], [-1.0, 0.0], [0.0, 1.0]]])
  s_out = {
    'encoded': encoded_proj,
    'attn_logits': torch.tensor([[1.0, 0.0, 100.0, -100.0]]),
    'mask': torch.tensor([[True, True, False, False]]),
  }
  target = torch.nn.functional.cosine_similarity(
    encoded_proj,
    t_out['hidden'].unsqueeze(1),
    dim=-1,
  )
  expected = ((1.0 - target[0, 0]) ** 2).reshape(())

  torch.testing.assert_close(loss(s_out, t_out, labels), expected)


def test_topk_cosine_attention_logit_loss_keeps_at_least_one_patch():
  loss = TopKCosineAttentionLogitLoss(topk_ratio=0.25)
  labels = torch.zeros(1)
  t_out = {'hidden': torch.tensor([[1.0, 0.0]])}
  encoded_proj = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]])
  s_out = {
    'encoded': encoded_proj,
    'attn_logits': torch.tensor([[4.0, -3.0, 9.0]]),
    'mask': torch.tensor([[False, True, False]]),
  }
  expected = ((-3.0 - 1.0) ** 2) * torch.ones(())

  torch.testing.assert_close(loss(s_out, t_out, labels), expected)


def test_tga_condition_name_supports_topk_cosine_term_slug():
  distill_loss = CompositeDistillationLoss([
    TaskLoss(),
    TopKCosineAttentionLogitLoss(topk_ratio=0.25),
  ])

  assert build_condition_name('teacher_guided_attention', distill_loss) == (
    'teacher_guided_attention_task_attn_cosine_topk_r0p25'
  )


def test_similarity_preserving_loss_describe_and_slug():
  loss = SimilarityPreservingLoss()

  assert loss.describe() == 'L_similarity_preserving'
  assert loss.slug() == 'similarity_preserving'


def test_similarity_preserving_loss_returns_zero_for_b1():
  loss = SimilarityPreservingLoss()
  labels = torch.zeros(1)
  s_out = {'hidden': torch.tensor([[1.0, -1.0]])}
  t_out = {'hidden': torch.tensor([[0.5, 0.5]])}

  torch.testing.assert_close(loss(s_out, t_out, labels), torch.zeros(()))


def test_similarity_preserving_loss_near_zero_for_identical_hidden():
  loss = SimilarityPreservingLoss()
  labels = torch.zeros(4)
  hidden = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.5, 0.5],
  ])
  s_out = {'hidden': hidden.clone().requires_grad_(True)}
  t_out = {'hidden': hidden.clone()}

  value = loss(s_out, t_out, labels)
  assert torch.isfinite(value)
  torch.testing.assert_close(value, torch.zeros(()), atol=1e-7, rtol=0.0)
  value.backward()
  assert torch.isfinite(s_out['hidden'].grad).all()


def test_decoupled_knowledge_distillation_loss_describe_and_slug():
  loss = DecoupledKnowledgeDistillationLoss(temperature=4.0, alpha=1.0, beta=4.0)

  assert loss.describe() == 'L_dkd(T=4, alpha=1, beta=4)'
  assert loss.slug() == 'dkd_t4p0_a1p0_b4p0'


def test_decoupled_knowledge_distillation_loss_near_zero_for_identical_logits():
  loss = DecoupledKnowledgeDistillationLoss(temperature=4.0, alpha=1.0, beta=4.0)
  labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
  shared_logits = torch.tensor([[0.2], [-1.1], [1.4], [-0.3]])
  s_out = {'logits': shared_logits.clone().requires_grad_(True)}
  t_out = {'logit': shared_logits.clone()}

  value = loss(s_out, t_out, labels)
  assert torch.isfinite(value)
  torch.testing.assert_close(value, torch.zeros(()), atol=1e-7, rtol=0.0)
  value.backward()
  assert torch.isfinite(s_out['logits'].grad).all()


def test_decoupled_knowledge_distillation_loss_is_finite_on_random_logits():
  loss = DecoupledKnowledgeDistillationLoss(temperature=4.0, alpha=1.0, beta=4.0)
  labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
  s_out = {'logits': torch.tensor([[2.0], [-3.0], [0.5], [1.2], [-0.7]], requires_grad=True)}
  t_out = {'logit': torch.tensor([[1.5], [-1.0], [0.2], [0.8], [0.4]])}

  value = loss(s_out, t_out, labels)
  assert torch.isfinite(value)
  value.backward()
  assert torch.isfinite(s_out['logits'].grad).all()


def test_decoupled_knowledge_distillation_binary_nckd_degenerates_to_zero():
  loss = DecoupledKnowledgeDistillationLoss(temperature=4.0, alpha=0.0, beta=1.0)
  labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
  s_out = {'logits': torch.tensor([[2.5], [-0.2], [1.1], [-3.4]])}
  t_out = {'logit': torch.tensor([[-1.7], [0.4], [-2.0], [0.9]])}

  value = loss(s_out, t_out, labels)
  torch.testing.assert_close(value, torch.zeros(()), atol=1e-7, rtol=0.0)


def test_soft_distribution_attention_loss_describe_and_slug():
  detached = SoftDistributionAttentionLoss(detach_target_encoded=True)
  coupled = SoftDistributionAttentionLoss(detach_target_encoded=False)
  no_zscore = SoftDistributionAttentionLoss(
    detach_target_encoded=True,
    normalize_target=False,
  )

  assert detached.describe() == 'L_attn_soft_distribution_detach'
  assert detached.slug() == 'attn_soft_distribution_detach'
  assert coupled.describe() == 'L_attn_soft_distribution_no_detach'
  assert coupled.slug() == 'attn_soft_distribution_no_detach'
  assert no_zscore.describe() == 'L_attn_soft_distribution_detach(no_zscore)'
  assert no_zscore.slug() == 'attn_soft_distribution_detach_no_zscore'


def test_soft_distribution_attention_loss_zero_for_uniform_constant_bag():
  loss = SoftDistributionAttentionLoss(detach_target_encoded=True)
  labels = torch.zeros(1)
  s_out = {
    'encoded': torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], requires_grad=True),
    'attn_logits': torch.tensor([[0.0, 0.0]], requires_grad=True),
  }
  t_out = {'hidden': torch.tensor([[1.0, 0.0]])}

  torch.testing.assert_close(loss(s_out, t_out, labels), torch.zeros(()))


def test_soft_distribution_attention_loss_respects_mask_and_stays_finite():
  loss = SoftDistributionAttentionLoss(detach_target_encoded=True)
  labels = torch.zeros(2)
  s_out = {
    'encoded': torch.tensor([
      [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
      [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    ], requires_grad=True),
    'attn_logits': torch.tensor([
      [3.0, -2.0, 100.0],
      [0.0, 0.0, 1.0],
    ], requires_grad=True),
    'mask': torch.tensor([
      [True, True, False],
      [True, False, False],
    ]),
  }
  t_out = {'hidden': torch.tensor([[1.0, 0.0], [1.0, 0.0]])}

  value = loss(s_out, t_out, labels)
  assert torch.isfinite(value)
  value.backward()
  assert torch.isfinite(s_out['attn_logits'].grad).all()


def test_tga_detach_controls_encoded_gradient_path():
  labels = torch.zeros(1)
  t_out = {'hidden': torch.tensor([[1.0, 0.0]])}

  detached_s_out = {
    'encoded': torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True),
    'attn_logits': torch.tensor([[2.0, -1.0]], requires_grad=True),
  }
  detached_loss = SoftDistributionAttentionLoss(detach_target_encoded=True)
  detached_value = detached_loss(detached_s_out, t_out, labels)
  detached_value.backward()
  assert detached_s_out['encoded'].grad is None
  assert detached_s_out['attn_logits'].grad is not None

  coupled_s_out = {
    'encoded': torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True),
    'attn_logits': torch.tensor([[2.0, -1.0]], requires_grad=True),
  }
  coupled_loss = SoftDistributionAttentionLoss(detach_target_encoded=False)
  coupled_value = coupled_loss(coupled_s_out, t_out, labels)
  coupled_value.backward()
  assert coupled_s_out['encoded'].grad is not None
  assert torch.isfinite(coupled_s_out['encoded'].grad).all()


def test_batch_contrastive_attention_loss_describe_and_slug():
  detached = BatchContrastiveAttentionLoss(detach_target_encoded=True)
  coupled = BatchContrastiveAttentionLoss(detach_target_encoded=False)
  tuned = BatchContrastiveAttentionLoss(
    detach_target_encoded=True,
    tau_neg=1.0,
    normalize_delta=False,
  )

  assert detached.describe() == 'L_attn_batch_contrastive_detach'
  assert detached.slug() == 'attn_batch_contrastive_detach'
  assert coupled.describe() == 'L_attn_batch_contrastive_no_detach'
  assert coupled.slug() == 'attn_batch_contrastive_no_detach'
  assert tuned.describe() == 'L_attn_batch_contrastive_detach(Tn=1, no_zscore)'
  assert tuned.slug() == 'attn_batch_contrastive_detach_tn1p0_no_zscore'


def test_batch_contrastive_attention_loss_handles_b1_without_nan():
  loss = BatchContrastiveAttentionLoss(detach_target_encoded=True)
  labels = torch.zeros(1)
  s_out = {
    'encoded': torch.tensor([[[1.0, 0.0], [0.5, 0.5]]], requires_grad=True),
    'attn_logits': torch.tensor([[0.0, 0.0]], requires_grad=True),
  }
  t_out = {'hidden': torch.tensor([[1.0, 0.0]])}

  value = loss(s_out, t_out, labels)
  assert torch.isfinite(value)
  torch.testing.assert_close(value, torch.zeros(()))


def test_batch_contrastive_attention_loss_respects_mask_and_is_finite():
  loss = BatchContrastiveAttentionLoss(detach_target_encoded=True)
  labels = torch.zeros(2)
  s_out = {
    'encoded': torch.tensor([
      [[1.0, 0.0], [0.2, 0.8], [0.5, 0.5]],
      [[0.0, 1.0], [0.7, 0.3], [1.0, 0.0]],
    ], requires_grad=True),
    'attn_logits': torch.tensor([
      [2.0, -1.0, 50.0],
      [-1.5, 1.5, -50.0],
    ], requires_grad=True),
    'mask': torch.tensor([
      [True, True, False],
      [True, False, False],
    ]),
  }
  t_out = {'hidden': torch.tensor([[1.0, 0.0], [0.0, 1.0]])}

  value = loss(s_out, t_out, labels)
  assert torch.isfinite(value)
  value.backward()
  assert torch.isfinite(s_out['attn_logits'].grad).all()


def test_batch_contrastive_detach_controls_encoded_gradient_path():
  labels = torch.zeros(2)
  t_out = {'hidden': torch.tensor([[1.0, 0.0], [0.0, 1.0]])}

  detached_s_out = {
    'encoded': torch.tensor([
      [[1.0, 0.0], [0.0, 1.0]],
      [[0.8, 0.2], [0.2, 0.8]],
    ], requires_grad=True),
    'attn_logits': torch.tensor([
      [1.0, -1.0],
      [0.5, -0.5],
    ], requires_grad=True),
  }
  detached_loss = BatchContrastiveAttentionLoss(detach_target_encoded=True)
  detached_value = detached_loss(detached_s_out, t_out, labels)
  detached_value.backward()
  assert detached_s_out['encoded'].grad is None
  assert detached_s_out['attn_logits'].grad is not None

  coupled_s_out = {
    'encoded': torch.tensor([
      [[1.0, 0.0], [0.0, 1.0]],
      [[0.8, 0.2], [0.2, 0.8]],
    ], requires_grad=True),
    'attn_logits': torch.tensor([
      [1.0, -1.0],
      [0.5, -0.5],
    ], requires_grad=True),
  }
  coupled_loss = BatchContrastiveAttentionLoss(detach_target_encoded=False)
  coupled_value = coupled_loss(coupled_s_out, t_out, labels)
  coupled_value.backward()
  assert coupled_s_out['encoded'].grad is not None
  assert torch.isfinite(coupled_s_out['encoded'].grad).all()
