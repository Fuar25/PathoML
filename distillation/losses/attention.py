"""Deprecated family wrappers for attention-guided distillation."""

from __future__ import annotations

from .base import CompositeDistillationLoss, WeightedTerm
from .terms import (
  ContrastiveTeacherDiscriminationLoss,
  CosineAttentionLogitLoss,
  DiscriminationAttentionLogitLoss,
  HiddenLoss,
  SoftLabelLoss,
  TaskLoss,
)


class TeacherGuidedAttnLoss(CompositeDistillationLoss):
  """Compatibility wrapper over explicit teacher-guided attention terms."""

  def __init__(
    self,
    alpha: float = 1.0,
    beta: float = 1.0,
    temperature: float = 4.0,
    gamma: float = 1.0,
    delta: float = 0.0,
    tau: float = 1.0,
  ) -> None:
    if delta != 0:
      raise NotImplementedError(
        "delta/mean-bypass is not part of the current compositional loss API."
      )

    self.alpha = float(alpha)
    self.beta = float(beta)
    self.temperature = float(temperature)
    self.gamma = float(gamma)
    self.delta = float(delta)
    self.tau = float(tau)

    terms = [TaskLoss()]
    if self.alpha != 0:
      terms.append(WeightedTerm(HiddenLoss(), self.alpha))
    if self.beta != 0:
      terms.append(WeightedTerm(SoftLabelLoss(self.temperature), self.beta))
    if self.gamma != 0:
      terms.append(WeightedTerm(CosineAttentionLogitLoss(), self.gamma))
    super().__init__(terms)

  def __repr__(self) -> str:
    return (
      f"TeacherGuidedAttnLoss(alpha={self.alpha}, beta={self.beta}, "
      f"temperature={self.temperature}, gamma={self.gamma}, "
      f"delta={self.delta}, tau={self.tau})"
    )


class RelationalTGALoss(CompositeDistillationLoss):
  """Compatibility wrapper over explicit relational TGA terms."""

  def __init__(
    self,
    gamma: float = 1.0,
    lam: float = 1.0,
    tau: float = 1.0,
  ) -> None:
    self.gamma = float(gamma)
    self.lam = float(lam)
    self.tau = float(tau)

    terms = [TaskLoss()]
    if self.gamma != 0:
      terms.append(WeightedTerm(DiscriminationAttentionLogitLoss(), self.gamma))
    if self.lam != 0:
      terms.append(
        WeightedTerm(ContrastiveTeacherDiscriminationLoss(self.tau), self.lam)
      )
    super().__init__(terms)

  def __repr__(self) -> str:
    return f"RelationalTGALoss(gamma={self.gamma}, lam={self.lam}, tau={self.tau})"
