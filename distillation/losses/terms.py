"""Atomic distillation loss terms."""

from __future__ import annotations

import math

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


def _masked_mse_per_bag(pred: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
  """Compute one MSE value per bag over valid instances."""
  sq_diff = (pred - target) ** 2
  if mask is None:
    return sq_diff.mean(dim=1)

  valid = mask.float()
  return (sq_diff * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)


def _teacher_confidence_gate(
  t_out: dict,
  *,
  min_confidence: float = 0.0,
) -> Tensor:
  """Return a continuous teacher-confidence gate in [0, 1]."""
  probs = torch.sigmoid(t_out['logit'].view(-1))
  confidence = 2.0 * (probs - 0.5).abs()
  if min_confidence <= 0.0:
    return confidence
  if min_confidence >= 1.0:
    return torch.zeros_like(confidence)
  return ((confidence - min_confidence).clamp_min(0.0) / (1.0 - min_confidence))


def _teacher_guided_cosine_target(
  s_out: dict,
  t_out: dict,
  *,
  detach_target_encoded: bool = False,
) -> Tensor:
  """Build patch-wise cosine scores against the teacher bag representation."""
  encoded = s_out['encoded'].detach() if detach_target_encoded else s_out['encoded']
  return F.cosine_similarity(
    encoded,
    t_out['hidden'].unsqueeze(1),
    dim=-1,
  )


def _class_aware_cosine_target(
  s_out: dict,
  t_out: dict,
  labels: Tensor,
  *,
  hidden_weight: float,
  class_weight: float,
  detach_target_encoded: bool = False,
) -> Tensor:
  """Build patch scores from teacher hidden and signed class direction."""
  encoded = s_out['encoded'].detach() if detach_target_encoded else s_out['encoded']
  target = encoded.new_zeros(encoded.size(0), encoded.size(1))

  if hidden_weight != 0.0:
    hidden_scores = F.cosine_similarity(
      encoded,
      t_out['hidden'].unsqueeze(1),
      dim=-1,
    )
    target = target + hidden_weight * hidden_scores

  if class_weight != 0.0:
    direction = t_out['class_weight'].view(1, -1).expand(encoded.size(0), -1)
    sign = torch.where(
      labels.view(-1, 1) > 0.5,
      torch.ones_like(direction),
      -torch.ones_like(direction),
    )
    signed_direction = direction * sign
    class_scores = F.cosine_similarity(
      encoded,
      signed_direction.unsqueeze(1),
      dim=-1,
    )
    target = target + class_weight * class_scores

  return target


def _masked_zscore(
  values: Tensor,
  mask: Tensor | None,
  *,
  eps: float = 1e-6,
) -> Tensor:
  """Z-score valid values within each bag; constant bags collapse to zeros."""
  normalized = torch.zeros_like(values)

  for bag_idx in range(values.size(0)):
    if mask is None:
      valid_idx = torch.arange(values.size(1), device=values.device)
    else:
      valid_idx = torch.nonzero(mask[bag_idx], as_tuple=False).flatten()
    if valid_idx.numel() == 0:
      continue

    valid_values = values[bag_idx, valid_idx]
    std = valid_values.std(unbiased=False)
    if valid_idx.numel() < 2 or std <= eps:
      normalized[bag_idx, valid_idx] = 0.0
      continue

    mean = valid_values.mean()
    normalized[bag_idx, valid_idx] = (valid_values - mean) / std

  return normalized


def _masked_softmax(logits: Tensor, mask: Tensor | None) -> Tensor:
  """Softmax only over valid instances within each bag."""
  if mask is None:
    return torch.softmax(logits, dim=1)

  probs = torch.zeros_like(logits)
  for bag_idx in range(logits.size(0)):
    valid_idx = torch.nonzero(mask[bag_idx], as_tuple=False).flatten()
    if valid_idx.numel() == 0:
      continue
    probs[bag_idx, valid_idx] = torch.softmax(logits[bag_idx, valid_idx], dim=0)
  return probs


def _masked_kl_div(
  target_probs: Tensor,
  student_probs: Tensor,
  mask: Tensor | None,
  *,
  eps: float = 1e-8,
) -> Tensor:
  """Average KL(target || student) over valid bags."""
  batch_losses: list[Tensor] = []

  for bag_idx in range(target_probs.size(0)):
    if mask is None:
      valid_idx = torch.arange(target_probs.size(1), device=target_probs.device)
    else:
      valid_idx = torch.nonzero(mask[bag_idx], as_tuple=False).flatten()
    if valid_idx.numel() == 0:
      continue

    target = target_probs[bag_idx, valid_idx].clamp_min(eps)
    student = student_probs[bag_idx, valid_idx].clamp_min(eps)
    batch_losses.append((target * (target.log() - student.log())).sum())

  if not batch_losses:
    return target_probs.new_zeros(())
  return torch.stack(batch_losses).mean()


def _batch_contrastive_delta(
  encoded: Tensor,
  teacher_hidden: Tensor,
  *,
  tau_neg: float,
) -> Tensor:
  """Compute per-patch positive-vs-negative contrastive deltas within a batch."""
  batch_size = encoded.size(0)
  if batch_size <= 1:
    return encoded.new_zeros(encoded.size(0), encoded.size(1))

  encoded_norm = F.normalize(encoded, p=2, dim=-1)
  teacher_norm = F.normalize(teacher_hidden, p=2, dim=-1)
  sim_all = torch.einsum('bnd,md->bnm', encoded_norm, teacher_norm)  # (B, N, B)

  batch_idx = torch.arange(batch_size, device=encoded.device)
  pos_scores = sim_all[batch_idx, :, batch_idx]  # (B, N)

  neg_logits = sim_all / tau_neg
  neg_logits = neg_logits.clone()
  neg_logits[batch_idx, :, batch_idx] = float('-inf')
  neg_logsumexp = torch.logsumexp(neg_logits, dim=2)

  delta = pos_scores - tau_neg * neg_logsumexp
  return torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))



def _masked_pairwise_rank_loss(
  pred: Tensor,
  target: Tensor,
  mask: Tensor | None,
  *,
  target_eps: float = 1e-8,
) -> Tensor:
  """Compute pairwise ranking loss over valid instance pairs within each bag."""
  batch_losses: list[Tensor] = []
  device = pred.device

  for bag_idx in range(pred.size(0)):
    bag_mask = mask[bag_idx] if mask is not None else None
    if bag_mask is None:
      valid_pred = pred[bag_idx]
      valid_target = target[bag_idx]
    else:
      valid_pred = pred[bag_idx][bag_mask]
      valid_target = target[bag_idx][bag_mask]

    if valid_pred.numel() < 2:
      continue

    pred_delta = valid_pred.unsqueeze(1) - valid_pred.unsqueeze(0)
    target_delta = valid_target.unsqueeze(1) - valid_target.unsqueeze(0)
    pair_mask = torch.triu(
      torch.ones_like(target_delta, dtype=torch.bool, device=device),
      diagonal=1,
    )
    pair_mask &= target_delta.abs() > target_eps
    if not pair_mask.any():
      continue

    signed_pred_delta = torch.sign(target_delta[pair_mask]) * pred_delta[pair_mask]
    batch_losses.append(F.softplus(-signed_pred_delta).mean())

  if not batch_losses:
    return pred.new_zeros(())
  return torch.stack(batch_losses).mean()


def _topk_teacher_mask(
  target: Tensor,
  mask: Tensor | None,
  *,
  topk_ratio: float,
) -> Tensor:
  """Return a boolean mask for top-k valid teacher targets within each bag."""
  topk_mask = torch.zeros_like(target, dtype=torch.bool)
  for bag_idx in range(target.size(0)):
    if mask is None:
      valid_idx = torch.arange(target.size(1), device=target.device)
    else:
      valid_idx = torch.nonzero(mask[bag_idx], as_tuple=False).flatten()
    if valid_idx.numel() == 0:
      continue
    k = max(1, math.ceil(valid_idx.numel() * topk_ratio))
    valid_target = target[bag_idx, valid_idx]
    topk_local = torch.topk(valid_target, k=k, largest=True).indices
    topk_mask[bag_idx, valid_idx[topk_local]] = True
  return topk_mask


def _masked_top_bottom_margin_loss(
  pred: Tensor,
  target: Tensor,
  mask: Tensor | None,
  *,
  top_ratio: float,
  margin: float,
) -> Tensor:
  """Rank teacher-selected top patches above bottom patches within each bag."""
  batch_losses: list[Tensor] = []

  for bag_idx in range(pred.size(0)):
    if mask is None:
      valid_idx = torch.arange(pred.size(1), device=pred.device)
    else:
      valid_idx = torch.nonzero(mask[bag_idx], as_tuple=False).flatten()
    if valid_idx.numel() < 2:
      continue

    k = max(1, math.ceil(valid_idx.numel() * top_ratio))
    k = min(k, valid_idx.numel() // 2)
    valid_target = target[bag_idx, valid_idx]
    valid_pred = pred[bag_idx, valid_idx]

    top_local = torch.topk(valid_target, k=k, largest=True).indices
    bottom_local = torch.topk(valid_target, k=k, largest=False).indices
    top_mean = valid_pred[top_local].mean()
    bottom_mean = valid_pred[bottom_local].mean()
    batch_losses.append(F.relu(pred.new_tensor(margin) - (top_mean - bottom_mean)))

  if not batch_losses:
    return pred.new_zeros(())
  return torch.stack(batch_losses).mean()


def _masked_topk_mse(
  pred: Tensor,
  target: Tensor,
  topk_mask: Tensor,
) -> Tensor:
  """Compute MSE over a preselected boolean top-k mask."""
  if not topk_mask.any():
    return pred.new_zeros(())
  sq_diff = (pred - target) ** 2
  return sq_diff[topk_mask].mean()


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
    return F.mse_loss(s_out['hidden'], t_out['hidden'])

  def describe(self) -> str:
    return "L_hidden"

  def slug(self) -> str:
    return "hidden"


class SimilarityPreservingLoss(DistillationTerm):
  """Match off-diagonal pairwise cosine-similarity structure within a batch."""

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    s_hidden = s_out['hidden']
    t_hidden = t_out['hidden']
    batch_size = s_hidden.size(0)
    if batch_size <= 1:
      return s_hidden.new_zeros(())

    s_norm = F.normalize(s_hidden, p=2, dim=1)
    t_norm = F.normalize(t_hidden, p=2, dim=1)
    s_sim = torch.mm(s_norm, s_norm.t())
    t_sim = torch.mm(t_norm, t_norm.t())

    off_diagonal_mask = ~torch.eye(batch_size, dtype=torch.bool, device=s_hidden.device)
    if not off_diagonal_mask.any():
      return s_hidden.new_zeros(())
    return F.mse_loss(s_sim[off_diagonal_mask], t_sim[off_diagonal_mask])

  def describe(self) -> str:
    return "L_similarity_preserving"

  def slug(self) -> str:
    return "similarity_preserving"


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


class DecoupledKnowledgeDistillationLoss(DistillationTerm):
  """Strict DKD over a binary single-logit head via two-logit recasting."""

  def __init__(
    self,
    temperature: float = 4.0,
    alpha: float = 1.0,
    beta: float = 4.0,
    eps: float = 1e-8,
  ) -> None:
    super().__init__()
    self.temperature = float(temperature)
    self.alpha = float(alpha)
    self.beta = float(beta)
    self.eps = float(eps)

  def _to_two_logits(self, logits: Tensor) -> Tensor:
    """Recast binary logit z into two-class logits [0, z]."""
    z = logits.squeeze(1) if logits.ndim == 2 else logits
    zeros = torch.zeros_like(z)
    return torch.stack((zeros, z), dim=1)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    y = labels.long()
    s_two = self._to_two_logits(s_out['logits'])
    t_two = self._to_two_logits(t_out['logit'])

    s_probs = torch.softmax(s_two / self.temperature, dim=1)
    t_probs = torch.softmax(t_two / self.temperature, dim=1)

    target_mask = F.one_hot(y, num_classes=2).bool()

    # TCKD: target class vs aggregated non-target probability.
    t_target = t_probs[target_mask].unsqueeze(1)
    s_target = s_probs[target_mask].unsqueeze(1)
    t_tckd = torch.cat((t_target, 1.0 - t_target), dim=1)
    s_tckd = torch.cat((s_target, 1.0 - s_target), dim=1)
    t_tckd = t_tckd.clamp_min(self.eps)
    s_tckd = s_tckd.clamp_min(self.eps)
    tckd = (t_tckd * (t_tckd.log() - s_tckd.log())).sum(dim=1).mean()

    # NCKD: KL over non-target class distribution (strict DKD form).
    neg_inf = float('-inf')
    t_nt_probs = torch.softmax((t_two / self.temperature).masked_fill(target_mask, neg_inf), dim=1)
    s_nt_probs = torch.softmax((s_two / self.temperature).masked_fill(target_mask, neg_inf), dim=1)
    t_nt_probs = t_nt_probs.clamp_min(self.eps)
    s_nt_probs = s_nt_probs.clamp_min(self.eps)
    nckd = (t_nt_probs * (t_nt_probs.log() - s_nt_probs.log())).sum(dim=1).mean()

    return (self.temperature ** 2) * (self.alpha * tckd + self.beta * nckd)

  def describe(self) -> str:
    return (
      "L_dkd("
      f"T={format_formula_value(self.temperature)}, "
      f"alpha={format_formula_value(self.alpha)}, "
      f"beta={format_formula_value(self.beta)})"
    )

  def slug(self) -> str:
    return (
      "dkd_"
      f"t{format_slug_value(self.temperature)}_"
      f"a{format_slug_value(self.alpha)}_"
      f"b{format_slug_value(self.beta)}"
    )


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
    return _rkd_distance(s_out['hidden'], t_out['hidden'], self.eps)

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
    return _rkd_angle(s_out['hidden'], t_out['hidden'])

  def describe(self) -> str:
    return "L_rkd_angle"

  def slug(self) -> str:
    return "rkd_angle"


class CosineAttentionLogitLoss(DistillationTerm):
  """Logit-space teacher-guided cosine attention supervision."""

  def __init__(self, detach_target_encoded: bool = False) -> None:
    super().__init__()
    self.detach_target_encoded = bool(detach_target_encoded)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    target = _teacher_guided_cosine_target(
      s_out,
      t_out,
      detach_target_encoded=self.detach_target_encoded,
    )
    return _masked_mse(s_out['attn_logits'], target, s_out.get('mask'))

  def describe(self) -> str:
    if self.detach_target_encoded:
      return "L_attn_cosine_detach"
    return "L_attn_cosine_no_detach"

  def slug(self) -> str:
    if self.detach_target_encoded:
      return "attn_cosine_logits_detach"
    return "attn_cosine_logits_no_detach"


class ConfidenceGatedCosineAttentionLogitLoss(DistillationTerm):
  """Cosine-logit TGA weighted by teacher prediction confidence."""

  def __init__(
    self,
    *,
    detach_target_encoded: bool = False,
    min_confidence: float = 0.0,
    normalize_by_gate: bool = False,
  ) -> None:
    super().__init__()
    self.detach_target_encoded = bool(detach_target_encoded)
    self.min_confidence = float(min_confidence)
    self.normalize_by_gate = bool(normalize_by_gate)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    target = _teacher_guided_cosine_target(
      s_out,
      t_out,
      detach_target_encoded=self.detach_target_encoded,
    )
    per_bag_loss = _masked_mse_per_bag(
      s_out['attn_logits'],
      target,
      s_out.get('mask'),
    )
    gates = _teacher_confidence_gate(
      t_out,
      min_confidence=self.min_confidence,
    )
    if self.normalize_by_gate:
      return (per_bag_loss * gates).sum() / gates.sum().clamp(min=1e-8)
    return (per_bag_loss * gates).mean()

  def describe(self) -> str:
    mode = "confidence_gated_normalized" if self.normalize_by_gate else "confidence_gated"
    base = (
      f"L_attn_cosine_{mode}_detach"
      if self.detach_target_encoded
      else f"L_attn_cosine_{mode}_no_detach"
    )
    if self.min_confidence > 0.0:
      return f"{base}(min_confidence={format_formula_value(self.min_confidence)})"
    return base

  def slug(self) -> str:
    mode = "confidence_gated_normalized" if self.normalize_by_gate else "confidence_gated"
    parts = [
      f"attn_cosine_{mode}_detach"
      if self.detach_target_encoded
      else f"attn_cosine_{mode}_no_detach"
    ]
    if self.min_confidence > 0.0:
      parts.append(f"min_confidence{format_slug_value(self.min_confidence)}")
    return "_".join(parts)


class ClassAwareCosineAttentionLogitLoss(DistillationTerm):
  """Cosine-logit TGA with teacher hidden and signed classifier direction."""

  def __init__(
    self,
    *,
    hidden_weight: float = 0.5,
    class_weight: float = 0.5,
    detach_target_encoded: bool = False,
  ) -> None:
    super().__init__()
    self.hidden_weight = float(hidden_weight)
    self.class_weight = float(class_weight)
    self.detach_target_encoded = bool(detach_target_encoded)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    target = _class_aware_cosine_target(
      s_out,
      t_out,
      labels,
      hidden_weight=self.hidden_weight,
      class_weight=self.class_weight,
      detach_target_encoded=self.detach_target_encoded,
    )
    return _masked_mse(s_out['attn_logits'], target, s_out.get('mask'))

  def describe(self) -> str:
    base = (
      "L_attn_class_aware_cosine_detach"
      if self.detach_target_encoded
      else "L_attn_class_aware_cosine_no_detach"
    )
    return (
      f"{base}("
      f"h={format_formula_value(self.hidden_weight)}, "
      f"c={format_formula_value(self.class_weight)})"
    )

  def slug(self) -> str:
    detach = "detach" if self.detach_target_encoded else "no_detach"
    return (
      "attn_class_aware_cosine_"
      f"h{format_slug_value(self.hidden_weight)}_"
      f"c{format_slug_value(self.class_weight)}_"
      f"{detach}"
    )


class ClassAwareAttentionRankMarginLoss(DistillationTerm):
  """Class-aware TGA that ranks teacher-selected top patches over bottom patches."""

  def __init__(
    self,
    *,
    hidden_weight: float = 0.5,
    class_weight: float = 0.5,
    top_ratio: float = 0.25,
    margin: float = 1.0,
    detach_target_encoded: bool = True,
  ) -> None:
    super().__init__()
    if not 0.0 < top_ratio <= 0.5:
      raise ValueError("top_ratio must be in (0, 0.5].")
    self.hidden_weight = float(hidden_weight)
    self.class_weight = float(class_weight)
    self.top_ratio = float(top_ratio)
    self.margin = float(margin)
    self.detach_target_encoded = bool(detach_target_encoded)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    target = _class_aware_cosine_target(
      s_out,
      t_out,
      labels,
      hidden_weight=self.hidden_weight,
      class_weight=self.class_weight,
      detach_target_encoded=self.detach_target_encoded,
    )
    return _masked_top_bottom_margin_loss(
      s_out['attn_logits'],
      target,
      s_out.get('mask'),
      top_ratio=self.top_ratio,
      margin=self.margin,
    )

  def describe(self) -> str:
    base = (
      "L_attn_class_aware_rank_margin_detach"
      if self.detach_target_encoded
      else "L_attn_class_aware_rank_margin_no_detach"
    )
    return (
      f"{base}("
      f"h={format_formula_value(self.hidden_weight)}, "
      f"c={format_formula_value(self.class_weight)}, "
      f"r={format_formula_value(self.top_ratio)}, "
      f"m={format_formula_value(self.margin)})"
    )

  def slug(self) -> str:
    detach = "detach" if self.detach_target_encoded else "no_detach"
    return (
      "attn_class_aware_rank_margin_"
      f"h{format_slug_value(self.hidden_weight)}_"
      f"c{format_slug_value(self.class_weight)}_"
      f"r{format_slug_value(self.top_ratio)}_"
      f"m{format_slug_value(self.margin)}_"
      f"{detach}"
    )


class CosineAttentionRankLoss(DistillationTerm):
  """Pairwise rank supervision over cosine-derived teacher attention targets."""

  def __init__(self, target_eps: float = 1e-8) -> None:
    super().__init__()
    self.target_eps = float(target_eps)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    target = _teacher_guided_cosine_target(s_out, t_out)
    return _masked_pairwise_rank_loss(
      s_out['attn_logits'],
      target,
      s_out.get('mask'),
      target_eps=self.target_eps,
    )

  def describe(self) -> str:
    return "L_attn_cosine_rank"

  def slug(self) -> str:
    return "attn_cosine_rank"


class TopKCosineAttentionLogitLoss(DistillationTerm):
  """Cosine attention regression restricted to top-k teacher-selected patches."""

  def __init__(self, topk_ratio: float = 0.25) -> None:
    super().__init__()
    self.topk_ratio = float(topk_ratio)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    target = _teacher_guided_cosine_target(s_out, t_out)
    topk_mask = _topk_teacher_mask(
      target,
      s_out.get('mask'),
      topk_ratio=self.topk_ratio,
    )
    return _masked_topk_mse(s_out['attn_logits'], target, topk_mask)

  def describe(self) -> str:
    return f"L_attn_cosine_topk(r={self.topk_ratio:.2f})"

  def slug(self) -> str:
    return f"attn_cosine_topk_r{format_slug_value(self.topk_ratio)}"


class SoftDistributionAttentionLoss(DistillationTerm):
  """Teacher-guided bag-wise attention distribution matching."""

  def __init__(
    self,
    *,
    teacher_temperature: float = 1.0,
    student_temperature: float = 1.0,
    normalize_target: bool = True,
    detach_target_encoded: bool = True,
  ) -> None:
    super().__init__()
    self.teacher_temperature = float(teacher_temperature)
    self.student_temperature = float(student_temperature)
    self.normalize_target = bool(normalize_target)
    self.detach_target_encoded = bool(detach_target_encoded)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    mask = s_out.get('mask')
    target_scores = _teacher_guided_cosine_target(
      s_out,
      t_out,
      detach_target_encoded=self.detach_target_encoded,
    )
    if self.normalize_target:
      target_scores = _masked_zscore(target_scores, mask)
    target_probs = _masked_softmax(target_scores / self.teacher_temperature, mask)
    student_probs = _masked_softmax(s_out['attn_logits'] / self.student_temperature, mask)
    return _masked_kl_div(target_probs, student_probs, mask)

  def describe(self) -> str:
    base = (
      "L_attn_soft_distribution_detach"
      if self.detach_target_encoded
      else "L_attn_soft_distribution_no_detach"
    )
    extras: list[str] = []
    if self.teacher_temperature != 1.0:
      extras.append(f"Tt={format_formula_value(self.teacher_temperature)}")
    if self.student_temperature != 1.0:
      extras.append(f"Ts={format_formula_value(self.student_temperature)}")
    if not self.normalize_target:
      extras.append("no_zscore")
    if not extras:
      return base
    return f"{base}({', '.join(extras)})"

  def slug(self) -> str:
    parts = [
      "attn_soft_distribution_detach"
      if self.detach_target_encoded
      else "attn_soft_distribution_no_detach"
    ]
    if self.teacher_temperature != 1.0:
      parts.append(f"tt{format_slug_value(self.teacher_temperature)}")
    if self.student_temperature != 1.0:
      parts.append(f"ts{format_slug_value(self.student_temperature)}")
    if not self.normalize_target:
      parts.append("no_zscore")
    return "_".join(parts)


class BatchContrastiveAttentionLoss(DistillationTerm):
  """Batch-contrastive teacher-guided attention distribution matching."""

  def __init__(
    self,
    *,
    tau_neg: float = 0.5,
    tau_target: float = 1.0,
    tau_student: float = 1.0,
    normalize_delta: bool = True,
    detach_target_encoded: bool = True,
  ) -> None:
    super().__init__()
    self.tau_neg = float(tau_neg)
    self.tau_target = float(tau_target)
    self.tau_student = float(tau_student)
    self.normalize_delta = bool(normalize_delta)
    self.detach_target_encoded = bool(detach_target_encoded)

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    del labels
    mask = s_out.get('mask')
    encoded = s_out['encoded'].detach() if self.detach_target_encoded else s_out['encoded']
    delta = _batch_contrastive_delta(
      encoded,
      t_out['hidden'],
      tau_neg=self.tau_neg,
    )
    if self.normalize_delta:
      delta = _masked_zscore(delta, mask)
    target_probs = _masked_softmax(delta / self.tau_target, mask)
    student_probs = _masked_softmax(s_out['attn_logits'] / self.tau_student, mask)
    return _masked_kl_div(target_probs, student_probs, mask)

  def describe(self) -> str:
    base = (
      "L_attn_batch_contrastive_detach"
      if self.detach_target_encoded
      else "L_attn_batch_contrastive_no_detach"
    )
    extras: list[str] = []
    if self.tau_neg != 0.5:
      extras.append(f"Tn={format_formula_value(self.tau_neg)}")
    if self.tau_target != 1.0:
      extras.append(f"Tt={format_formula_value(self.tau_target)}")
    if self.tau_student != 1.0:
      extras.append(f"Ts={format_formula_value(self.tau_student)}")
    if not self.normalize_delta:
      extras.append("no_zscore")
    if not extras:
      return base
    return f"{base}({', '.join(extras)})"

  def slug(self) -> str:
    parts = [
      "attn_batch_contrastive_detach"
      if self.detach_target_encoded
      else "attn_batch_contrastive_no_detach"
    ]
    if self.tau_neg != 0.5:
      parts.append(f"tn{format_slug_value(self.tau_neg)}")
    if self.tau_target != 1.0:
      parts.append(f"tt{format_slug_value(self.tau_target)}")
    if self.tau_student != 1.0:
      parts.append(f"ts{format_slug_value(self.tau_student)}")
    if not self.normalize_delta:
      parts.append("no_zscore")
    return "_".join(parts)
