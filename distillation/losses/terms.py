"""Atomic distillation loss terms."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import DistillationTerm, format_formula_value, format_slug_value


def _masked_mse(pred: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
  """Compute MSE over the valid instance mask when provided."""
  if mask is None:
    return F.mse_loss(pred, target)
  sq_diff = ((pred - target) ** 2) * mask.float()
  return sq_diff.sum() / mask.float().sum().clamp(min=1)


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


class TaskLoss(DistillationTerm):
  """Base supervised task loss."""

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del t_out
    s_logit = s_out['logits'].squeeze(1)
    return F.binary_cross_entropy_with_logits(s_logit, labels)

  def describe(self) -> str:
    return "L_task"

  def slug(self) -> str:
    return "task"


class HiddenLoss(DistillationTerm):
  """Teacher hidden-state feature matching."""

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    s_hidden = s_out['hidden']
    s_feat = s_out.get('proj', s_hidden)
    return F.mse_loss(s_feat, t_out['hidden'])

  def describe(self) -> str:
    return "L_hidden"

  def slug(self) -> str:
    return "hidden"


class SoftLabelLoss(DistillationTerm):
  """Soft-label KD over teacher logits."""

  def __init__(self, temperature: float = 4.0) -> None:
    super().__init__()
    self.temperature = float(temperature)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    s_logit = s_out['logits'].squeeze(1)
    t_logit = t_out['logit'].squeeze(1)
    p_teacher = torch.sigmoid(t_logit / self.temperature)
    return F.binary_cross_entropy_with_logits(
      s_logit / self.temperature,
      p_teacher,
    ) * (self.temperature ** 2)

  def describe(self) -> str:
    return f"L_soft_label(T={format_formula_value(self.temperature)})"

  def slug(self) -> str:
    return f"soft_label_t{format_slug_value(self.temperature)}"


class RKDDistanceLoss(DistillationTerm):
  """Distance-wise relational KD."""

  def __init__(self, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    s_emb = s_out.get('proj', s_out['hidden'])
    return _rkd_distance(s_emb, t_out['hidden'], self.eps)

  def describe(self) -> str:
    return "L_rkd_distance"

  def slug(self) -> str:
    return "rkd_distance"


class RKDAngleLoss(DistillationTerm):
  """Angle-wise relational KD."""

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    s_emb = s_out.get('proj', s_out['hidden'])
    return _rkd_angle(s_emb, t_out['hidden'])

  def describe(self) -> str:
    return "L_rkd_angle"

  def slug(self) -> str:
    return "rkd_angle"


class CosineAttentionLogitLoss(DistillationTerm):
  """Logit-space teacher-guided cosine attention supervision."""

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    encoded_proj = s_out.get('encoded_proj', s_out['encoded'])
    target = F.cosine_similarity(
      encoded_proj.detach(),
      t_out['hidden'].unsqueeze(1),
      dim=-1,
    )
    return _masked_mse(s_out['attn_logits'], target, s_out.get('mask'))

  def describe(self) -> str:
    return "L_attn_cosine"

  def slug(self) -> str:
    return "attn_cosine_logits"


class DiscriminationAttentionLogitLoss(DistillationTerm):
  """Logit-space relational discrimination attention supervision."""

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    encoded_proj = s_out.get('encoded_proj', s_out['encoded'])
    sim_matrix = _patch_teacher_sim_matrix(encoded_proj, t_out['hidden'])
    target = _discrimination(sim_matrix).detach()
    return _masked_mse(s_out['attn_logits'], target, s_out.get('mask'))

  def describe(self) -> str:
    return "L_attn_discrimination"

  def slug(self) -> str:
    return "attn_discrimination"


class ContrastiveTeacherDiscriminationLoss(DistillationTerm):
  """Patch-to-teacher discrimination via contrastive supervision."""

  def __init__(self, tau: float = 1.0) -> None:
    super().__init__()
    self.tau = float(tau)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    encoded_proj = s_out.get('encoded_proj', s_out['encoded'])
    sim_matrix = _patch_teacher_sim_matrix(encoded_proj, t_out['hidden'])
    mask = s_out.get('mask')
    batch_size, n_instances, _ = encoded_proj.shape
    logits = sim_matrix / self.tau
    targets = torch.arange(batch_size, device=logits.device)
    targets = targets.unsqueeze(1).expand(batch_size, n_instances)
    if mask is not None:
      return F.cross_entropy(logits[mask], targets[mask])
    return F.cross_entropy(logits.reshape(-1, batch_size), targets.reshape(-1))

  def describe(self) -> str:
    if self.tau == 1.0:
      return "L_contrast"
    return f"L_contrast(T={format_formula_value(self.tau)})"

  def slug(self) -> str:
    if self.tau == 1.0:
      return "contrast"
    return f"contrast_t{format_slug_value(self.tau)}"
