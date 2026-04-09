"""Distillation loss package."""

from .base import (
  CompositeDistillationLoss,
  DistillationLoss,
  DistillationTerm,
  WeightedTerm,
)
from .terms import (
  ContrastiveTeacherDiscriminationLoss,
  CosineAttentionLogitLoss,
  DiscriminationAttentionLogitLoss,
  HiddenLoss,
  RKDAngleLoss,
  RKDDistanceLoss,
  SoftLabelLoss,
  TaskLoss,
)

__all__ = [
  'DistillationLoss',
  'DistillationTerm',
  'CompositeDistillationLoss',
  'WeightedTerm',
  'TaskLoss',
  'HiddenLoss',
  'SoftLabelLoss',
  'RKDDistanceLoss',
  'RKDAngleLoss',
  'CosineAttentionLogitLoss',
  'DiscriminationAttentionLogitLoss',
  'ContrastiveTeacherDiscriminationLoss',
]
