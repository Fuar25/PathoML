"""Standard teacher-student KD losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import DistillationLoss


class StandardKDLoss(DistillationLoss):
  """Task loss plus feature matching and softened logit KD."""

  def __init__(
    self,
    alpha: float = 1.0,
    beta: float = 1.0,
    temperature: float = 4.0,
  ) -> None:
    super().__init__()
    self.alpha = alpha
    self.beta = beta
    self.temperature = temperature

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    s_logit = s_out['logits'].squeeze(1)
    s_hidden = s_out['hidden']
    t_logit = t_out['logit'].squeeze(1)
    t_hidden = t_out['hidden']

    loss = F.binary_cross_entropy_with_logits(s_logit, labels)

    if self.alpha != 0:
      s_feat = s_out.get('proj', s_hidden)
      loss = loss + self.alpha * F.mse_loss(s_feat, t_hidden)

    if self.beta != 0:
      temperature = self.temperature
      p_teacher = torch.sigmoid(t_logit / temperature)
      kd_loss = F.binary_cross_entropy_with_logits(
        s_logit / temperature,
        p_teacher,
      ) * (temperature ** 2)
      loss = loss + self.beta * kd_loss

    return loss

  def __repr__(self) -> str:
    return (
      f"StandardKDLoss(alpha={self.alpha}, beta={self.beta}, "
      f"temperature={self.temperature})"
    )
