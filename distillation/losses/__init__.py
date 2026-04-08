"""Distillation loss package."""

from .attention import RelationalTGALoss, TeacherGuidedAttnLoss
from .base import DistillationLoss
from .relational import RKDLoss
from .standard import StandardKDLoss

__all__ = [
  'DistillationLoss',
  'StandardKDLoss',
  'RKDLoss',
  'TeacherGuidedAttnLoss',
  'RelationalTGALoss',
]
