"""Distillation loss package."""

from .base import (
  CompositeDistillationLoss,
  DistillationLoss,
  DistillationTerm,
  WeightedTerm,
)
from .terms import (
  BatchContrastiveAttentionLoss,
  ClassAwareAttentionRankMarginLoss,
  ClassAwareCosineAttentionLogitLoss,
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
)

__all__ = [
  'DistillationLoss',
  'DistillationTerm',
  'CompositeDistillationLoss',
  'WeightedTerm',
  'TaskLoss',
  'HiddenLoss',
  'SimilarityPreservingLoss',
  'SoftLabelLoss',
  'DecoupledKnowledgeDistillationLoss',
  'RKDDistanceLoss',
  'RKDAngleLoss',
  'BatchContrastiveAttentionLoss',
  'ClassAwareAttentionRankMarginLoss',
  'ClassAwareCosineAttentionLogitLoss',
  'ConfidenceGatedCosineAttentionLogitLoss',
  'CosineAttentionLogitLoss',
  'CosineAttentionRankLoss',
  'SoftDistributionAttentionLoss',
  'TopKCosineAttentionLogitLoss',
]
