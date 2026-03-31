"""蒸馏损失接口与实现。

DistillationLoss  — 所有蒸馏损失的抽象基类，定义统一的 forward 签名。
StandardKDLoss    — L_task + alpha * L_feat + beta * L_kd
RKDLoss           — L_task + alpha * L_feat + gamma * L_rkd（关系蒸馏）

扩展新方法：继承 DistillationLoss 并实现 forward 即可，无需改动 trainer。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# (1) 接口
# ---------------------------------------------------------------------------

class DistillationLoss(nn.Module, ABC):
  """蒸馏损失基类。所有蒸馏损失策略实现此接口。

  forward 签名:
    s_out:  student 输出 dict, 至少含 'logits' (B,1) 和 'hidden' (B, D_s)
    t_out:  teacher 输出 dict, 至少含 'logit' (B,1) 和 'hidden' (B, D_t)
    labels: ground truth (B,)
  返回:
    scalar loss tensor
  """

  @abstractmethod
  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    ...


# ---------------------------------------------------------------------------
# (2) StandardKDLoss — 从 trainer._compute_kd_loss 原封搬出
# ---------------------------------------------------------------------------

class StandardKDLoss(DistillationLoss):
  """L_total = L_task + alpha * L_feat + beta * L_kd

  - L_task: BCEWithLogitsLoss(s_logit, label)
  - L_feat: MSELoss(s_hidden, t_hidden)
  - L_kd:   BCEWithLogitsLoss(s_logit/T, sigmoid(t_logit/T)) * T²

  alpha=0, beta=0 时退化为纯 Baseline（仅 L_task）。
  """

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
    s_logit  = s_out['logits'].squeeze(1)   # (B,)
    s_hidden = s_out['hidden']              # (B, D_s)
    t_logit  = t_out['logit'].squeeze(1)    # (B,)，已 detach（no_grad）
    t_hidden = t_out['hidden']              # (B, D_t)，已 detach

    # (1) 任务损失
    loss = F.binary_cross_entropy_with_logits(s_logit, labels)

    # (2) 特征匹配损失（优先用 projection head 输出对齐 teacher）
    if self.alpha != 0:
      s_feat = s_out.get('proj', s_hidden)
      loss = loss + self.alpha * F.mse_loss(s_feat, t_hidden)

    # (3) Logit KD 损失
    if self.beta != 0:
      T = self.temperature
      p_teacher = torch.sigmoid(t_logit / T)
      L_kd = F.binary_cross_entropy_with_logits(
        s_logit / T, p_teacher
      ) * (T ** 2)
      loss = loss + self.beta * L_kd

    return loss

  def __repr__(self) -> str:
    return (
      f"StandardKDLoss(alpha={self.alpha}, beta={self.beta}, "
      f"temperature={self.temperature})"
    )


# ---------------------------------------------------------------------------
# (3) RKDLoss — Relational Knowledge Distillation (distance-wise)
# ---------------------------------------------------------------------------

class RKDLoss(DistillationLoss):
  """L_total = L_task + alpha * L_feat + gamma * L_rkd

  - L_task: BCEWithLogitsLoss(s_logit, label)
  - L_feat: MSELoss(s_proj, t_hidden)                         (alpha=0 禁用)
  - L_rkd:  smooth_l1(mu_norm(cdist(s)), mu_norm(cdist(t)))   (gamma=0 禁用)

  RKD 匹配样本间的距离结构而非单个表示，不依赖表示空间对齐。
  需要 batch_size > 1；B=1 时 L_rkd 自动退化为 0。
  """

  def __init__(
    self,
    alpha: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-6,
  ) -> None:
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.eps = eps

  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    s_logit  = s_out['logits'].squeeze(1)   # (B,)
    t_hidden = t_out['hidden']              # (B, D_t)

    # (1) 任务损失
    loss = F.binary_cross_entropy_with_logits(s_logit, labels)

    # (2) 特征匹配损失
    if self.alpha != 0:
      s_feat = s_out.get('proj', s_out['hidden'])
      loss = loss + self.alpha * F.mse_loss(s_feat, t_hidden)

    # (3) RKD distance-wise 关系蒸馏
    if self.gamma != 0:
      s_emb = s_out.get('proj', s_out['hidden'])  # (B, D)
      t_emb = t_hidden                             # (B, D)
      s_dist = torch.cdist(s_emb, s_emb, p=2)     # (B, B)
      t_dist = torch.cdist(t_emb, t_emb, p=2)     # (B, B)
      # mu-normalization
      s_dist = s_dist / (s_dist.mean() + self.eps)
      t_dist = t_dist / (t_dist.mean() + self.eps)
      loss = loss + self.gamma * F.smooth_l1_loss(s_dist, t_dist)

    return loss

  def __repr__(self) -> str:
    return f"RKDLoss(alpha={self.alpha}, gamma={self.gamma})"
