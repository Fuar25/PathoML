"""Deprecated family wrapper for relational knowledge distillation."""

from __future__ import annotations

from .base import CompositeDistillationLoss, WeightedTerm
from .terms import RKDAngleLoss, RKDDistanceLoss, TaskLoss


class RKDLoss(CompositeDistillationLoss):
  """Compatibility wrapper over explicit relational KD terms."""

  def __init__(
    self,
    gamma_d: float = 1.0,
    gamma_a: float = 2.0,
    eps: float = 1e-6,
  ) -> None:
    self.gamma_d = float(gamma_d)
    self.gamma_a = float(gamma_a)
    self.eps = eps

    terms = [TaskLoss()]
    if self.gamma_d != 0:
      terms.append(WeightedTerm(RKDDistanceLoss(self.eps), self.gamma_d))
    if self.gamma_a != 0:
      terms.append(WeightedTerm(RKDAngleLoss(), self.gamma_a))
    super().__init__(terms)

  def __repr__(self) -> str:
    return f"RKDLoss(gamma_d={self.gamma_d}, gamma_a={self.gamma_a})"
