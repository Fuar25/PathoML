"""Attention-guided distillation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import DistillationLoss
from .standard import StandardKDLoss


def _masked_mean(x: Tensor, mask: Tensor | None) -> Tensor:
  """Masked mean over the instance dimension."""
  if mask is None:
    return x.mean(dim=1)
  mask_f = mask.float().unsqueeze(-1)
  return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)


class TeacherGuidedAttnLoss(StandardKDLoss):
  """StandardKD plus attention supervision and optional mean bypass."""

  def __init__(
    self,
    alpha: float = 1.0,
    beta: float = 1.0,
    temperature: float = 4.0,
    gamma: float = 1.0,
    delta: float = 0.0,
    tau: float = 1.0,
  ) -> None:
    super().__init__(alpha, beta, temperature)
    self.gamma = gamma
    self.delta = delta
    self.tau = tau

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    loss = super().forward(s_out, t_out, labels)
    t_hidden = t_out['hidden']
    encoded_proj = s_out.get('encoded_proj', s_out['encoded'])
    mask = s_out.get('mask')

    if self.gamma != 0:
      attn_logits = s_out['attn_logits']
      target = F.cosine_similarity(
        encoded_proj.detach(),
        t_hidden.unsqueeze(1),
        dim=-1,
      )
      if mask is not None:
        sq_diff = ((attn_logits - target) ** 2) * mask.float()
        attn_loss = sq_diff.sum() / mask.float().sum().clamp(min=1)
      else:
        attn_loss = F.mse_loss(attn_logits, target)
      loss = loss + self.gamma * attn_loss

    if self.delta != 0:
      mean_proj = _masked_mean(encoded_proj, mask)
      loss = loss + self.delta * F.mse_loss(mean_proj, t_hidden)

    return loss

  def __repr__(self) -> str:
    return (
      f"TeacherGuidedAttnLoss(alpha={self.alpha}, beta={self.beta}, "
      f"temperature={self.temperature}, gamma={self.gamma}, "
      f"delta={self.delta}, tau={self.tau})"
    )


def _patch_teacher_sim_matrix(encoded_proj: Tensor, t_hidden: Tensor) -> Tensor:
  """Compute patch-to-teacher cosine similarity matrix."""
  patch_norm = F.normalize(encoded_proj, dim=-1)
  teacher_norm = F.normalize(t_hidden, dim=-1)
  return torch.einsum('bnd,jd->bnj', patch_norm, teacher_norm)


def _discrimination(sim_matrix: Tensor) -> Tensor:
  """Compute teacher discrimination scores for each patch."""
  batch_size = sim_matrix.size(0)
  batch_idx = torch.arange(batch_size, device=sim_matrix.device)
  own_sim = sim_matrix[batch_idx, :, batch_idx]
  diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
  others_sum = (sim_matrix * diag_mask.unsqueeze(1)).sum(dim=2)
  others_mean = others_sum / max(batch_size - 1, 1)
  return own_sim - others_mean


class RelationalTGALoss(DistillationLoss):
  """Task loss plus relational attention guidance and patch-level contrast."""

  def __init__(
    self,
    gamma: float = 1.0,
    lam: float = 1.0,
    tau: float = 1.0,
  ) -> None:
    super().__init__()
    self.gamma = gamma
    self.lam = lam
    self.tau = tau

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    s_logit = s_out['logits'].squeeze(1)
    t_hidden = t_out['hidden']
    encoded_proj = s_out.get('encoded_proj', s_out['encoded'])
    mask = s_out.get('mask')

    batch_size, n_instances, _ = encoded_proj.shape
    loss = F.binary_cross_entropy_with_logits(s_logit, labels)
    sim_matrix = _patch_teacher_sim_matrix(encoded_proj, t_hidden)

    if self.gamma != 0:
      attn_logits = s_out['attn_logits']
      target = _discrimination(sim_matrix).detach()
      if mask is not None:
        sq_diff = ((attn_logits - target) ** 2) * mask.float()
        n_valid = mask.float().sum().clamp(min=1)
        attn_loss = sq_diff.sum() / n_valid
      else:
        attn_loss = F.mse_loss(attn_logits, target)
      loss = loss + self.gamma * attn_loss

    if self.lam != 0:
      logits = sim_matrix / self.tau
      targets = torch.arange(batch_size, device=logits.device)
      targets = targets.unsqueeze(1).expand(batch_size, n_instances)
      if mask is not None:
        contrast_loss = F.cross_entropy(logits[mask], targets[mask])
      else:
        contrast_loss = F.cross_entropy(logits.reshape(-1, batch_size), targets.reshape(-1))
      loss = loss + self.lam * contrast_loss

    return loss

  def __repr__(self) -> str:
    return f"RelationalTGALoss(gamma={self.gamma}, lam={self.lam}, tau={self.tau})"
