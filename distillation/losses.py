"""蒸馏损失接口与实现。

DistillationLoss  — 所有蒸馏损失的抽象基类，定义统一的 forward 签名。
StandardKDLoss    — L_task + alpha * L_feat + beta * L_kd
RKDLoss           — L_task + gamma_d * L_dist + gamma_a * L_angle（关系蒸馏）

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
# (3) RKDLoss — Relational Knowledge Distillation (Park et al., CVPR 2019)
#     distance-wise (二元关系) + angle-wise (三元关系)
# ---------------------------------------------------------------------------

def _rkd_distance(s_emb: Tensor, t_emb: Tensor, eps: float = 1e-6) -> Tensor:
  """Distance-wise RKD: 匹配样本对之间的 L2 距离结构。"""
  s_dist = torch.cdist(s_emb, s_emb, p=2)       # (B, B)
  t_dist = torch.cdist(t_emb, t_emb, p=2)       # (B, B)
  # mu-normalization
  s_dist = s_dist / (s_dist.mean() + eps)
  t_dist = t_dist / (t_dist.mean() + eps)
  return F.smooth_l1_loss(s_dist, t_dist)


def _rkd_angle(s_emb: Tensor, t_emb: Tensor) -> Tensor:
  """Angle-wise RKD: 匹配三元组 (i, j, k) 中以 j 为顶点的角度结构。

  对每个中心点 j，计算所有 (i, k) 对构成的角度余弦值，
  要求 student 和 teacher 的角度结构一致。
  """
  # (1) 差向量: diff[j, i] = emb[i] - emb[j]，形状 (B, B, D)
  s_diff = s_emb.unsqueeze(0) - s_emb.unsqueeze(1)
  t_diff = t_emb.unsqueeze(0) - t_emb.unsqueeze(1)
  # (2) L2 归一化差向量
  s_diff = F.normalize(s_diff, p=2, dim=2)
  t_diff = F.normalize(t_diff, p=2, dim=2)
  # (3) 角度余弦: angle[j, i, k] = <s_diff[j,i], s_diff[j,k]>
  #     bmm 沿 j 维度做矩阵乘，得 (B, B, B)
  s_angle = torch.bmm(s_diff, s_diff.transpose(1, 2)).view(-1)
  t_angle = torch.bmm(t_diff, t_diff.transpose(1, 2)).view(-1)
  return F.smooth_l1_loss(s_angle, t_angle)


class RKDLoss(DistillationLoss):
  """L_total = L_task + gamma_d * L_dist + gamma_a * L_angle

  - L_task:  BCEWithLogitsLoss(s_logit, label)
  - L_dist:  distance-wise RKD — 匹配样本对间 L2 距离   (gamma_d=0 禁用)
  - L_angle: angle-wise RKD — 匹配三元组角度结构         (gamma_a=0 禁用)

  Ref: Park et al., "Relational Knowledge Distillation", CVPR 2019.
  需要 batch_size > 1；B=1 时两项自动退化为 0。
  """

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
    s_logit  = s_out['logits'].squeeze(1)   # (B,)
    t_hidden = t_out['hidden']              # (B, D_t)

    # (1) 任务损失
    loss = F.binary_cross_entropy_with_logits(s_logit, labels)

    # (2) RKD distance-wise
    if self.gamma_d != 0:
      s_emb = s_out.get('proj', s_out['hidden'])
      loss = loss + self.gamma_d * _rkd_distance(s_emb, t_hidden, self.eps)

    # (3) RKD angle-wise
    if self.gamma_a != 0:
      s_emb = s_out.get('proj', s_out['hidden'])
      loss = loss + self.gamma_a * _rkd_angle(s_emb, t_hidden)

    return loss

  def __repr__(self) -> str:
    return f"RKDLoss(gamma_d={self.gamma_d}, gamma_a={self.gamma_a})"
