"""Deprecated family wrapper for standard teacher-student KD."""

from __future__ import annotations

from .base import CompositeDistillationLoss, WeightedTerm
from .terms import HiddenLoss, SoftLabelLoss, TaskLoss


class StandardKDLoss(CompositeDistillationLoss):
  """Compatibility wrapper over explicit distillation terms."""

  def __init__(
    self,
    alpha: float = 1.0,
    beta: float = 1.0,
    temperature: float = 4.0,
  ) -> None:
    self.alpha = float(alpha)
    self.beta = float(beta)
    self.temperature = float(temperature)

    terms = [TaskLoss()]
    if self.alpha != 0:
      terms.append(WeightedTerm(HiddenLoss(), self.alpha))
    if self.beta != 0:
      terms.append(WeightedTerm(SoftLabelLoss(self.temperature), self.beta))
    super().__init__(terms)

  def __repr__(self) -> str:
    return (
      f"StandardKDLoss(alpha={self.alpha}, beta={self.beta}, "
      f"temperature={self.temperature})"
    )
