"""Relational knowledge distillation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import DistillationLoss


def _rkd_distance(s_emb: Tensor, t_emb: Tensor, eps: float = 1e-6) -> Tensor:
  """Match pairwise distance structure."""
  s_dist = torch.cdist(s_emb, s_emb, p=2)
  t_dist = torch.cdist(t_emb, t_emb, p=2)
  s_dist = s_dist / (s_dist.mean() + eps)
  t_dist = t_dist / (t_dist.mean() + eps)
  return F.smooth_l1_loss(s_dist, t_dist)


def _rkd_angle(s_emb: Tensor, t_emb: Tensor) -> Tensor:
  """Match triplet angle structure."""
  s_diff = s_emb.unsqueeze(0) - s_emb.unsqueeze(1)
  t_diff = t_emb.unsqueeze(0) - t_emb.unsqueeze(1)
  s_diff = F.normalize(s_diff, p=2, dim=2)
  t_diff = F.normalize(t_diff, p=2, dim=2)
  s_angle = torch.bmm(s_diff, s_diff.transpose(1, 2)).view(-1)
  t_angle = torch.bmm(t_diff, t_diff.transpose(1, 2)).view(-1)
  return F.smooth_l1_loss(s_angle, t_angle)


class RKDLoss(DistillationLoss):
  """Task loss plus distance-wise and angle-wise relational KD."""

  def __init__(
    self,
    gamma_d: float = 1.0,
    gamma_a: float = 2.0,
    eps: float = 1e-6,
  ) -> None:
    super().__init__()
    self.gamma_d = gamma_d
    self.gamma_a = gamma_a
    self.eps = eps

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    s_logit = s_out['logits'].squeeze(1)
    t_hidden = t_out['hidden']

    loss = F.binary_cross_entropy_with_logits(s_logit, labels)

    if self.gamma_d != 0:
      s_emb = s_out.get('proj', s_out['hidden'])
      loss = loss + self.gamma_d * _rkd_distance(s_emb, t_hidden, self.eps)

    if self.gamma_a != 0:
      s_emb = s_out.get('proj', s_out['hidden'])
      loss = loss + self.gamma_a * _rkd_angle(s_emb, t_hidden)

    return loss

  def __repr__(self) -> str:
    return f"RKDLoss(gamma_d={self.gamma_d}, gamma_a={self.gamma_a})"
