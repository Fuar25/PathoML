"""门控融合 MLP 分类模型，支持多模态自适应权重学习。"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import BaseModel, DataDict
from ..registry import register_model


@register_model('gated_fusion_mlp')
class GatedFusionMLP(BaseModel):
  """门控融合 MLP：每个模态学习一个自适应权重，加权融合后分类。

  将 concat 特征拆回各模态，独立投影到共享维度后，通过 gate network
  计算 softmax 归一化的模态权重，实现自适应融合。

  Usage:
      config.model.model_name = "gated_fusion_mlp"
      config.model.model_kwargs = {
          "n_modalities": 4,
          "hidden_dim": 256,
          "dropout": 0.2,
          "gate_temperature": 1.0,
      }
  """

  def __init__(
    self,
    input_dim: int,
    n_modalities: int,
    hidden_dim: int,
    num_classes: int = 1,
    dropout: float = 0.0,
    gate_hidden_dim: int | None = None,
    gate_temperature: float = 1.0,
    num_post_layers: int = 1,
    **kwargs,
  ) -> None:
    """初始化门控融合 MLP。

    Args:
      input_dim: 输入特征总维度（= n_modalities × D_per）。
      n_modalities: 模态数量。
      hidden_dim: 各模态投影维度 & 分类器隐藏层维度。
      num_classes: 输出类别数（二分类时为 1）。
      dropout: Dropout 比率。
      gate_hidden_dim: Gate network 隐藏层维度，默认等于 hidden_dim。
      gate_temperature: Softmax 温度，控制门控竞争强度。
      num_post_layers: 融合后分类器隐藏层数量。
      **kwargs: 忽略的额外参数。
    """
    super().__init__()
    assert input_dim % n_modalities == 0, (
      f"input_dim ({input_dim}) 必须能被 n_modalities ({n_modalities}) 整除"
    )

    self.n_modalities = n_modalities
    self.d_per = input_dim // n_modalities
    self.gate_temperature = gate_temperature
    gate_hidden_dim = gate_hidden_dim or hidden_dim
    drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # (1) 各模态独立投影: D_per → hidden_dim
    self.modality_projectors = nn.ModuleList([
      nn.Sequential(nn.Linear(self.d_per, hidden_dim), nn.GELU(), drop)
      for _ in range(n_modalities)
    ])

    # (2) Gate network: n_modalities * hidden_dim → n_modalities
    self.gate_net = nn.Sequential(
      nn.Linear(n_modalities * hidden_dim, gate_hidden_dim),
      nn.GELU(),
      drop,
      nn.Linear(gate_hidden_dim, n_modalities),
    )

    # (3) 融合后分类头
    cls_layers = []
    for _ in range(num_post_layers - 1):
      cls_layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), drop]
    cls_layers.append(nn.Linear(hidden_dim, num_classes))
    self.classifier = nn.Sequential(*cls_layers)

  def forward(self, data_dict: DataDict) -> Dict[str, torch.Tensor]:
    """门控融合前向传播。

    Args:
      data_dict: 输入字典，'features' 形状为 (B, D_concat) 或 (B, 1, D_concat)。

    Returns:
      包含 'logits' (B, num_classes) 和 'gates' (B, n_modalities) 的字典。
    """
    features = data_dict['features']
    if features.dim() == 3 and features.size(1) == 1:
      features = features.squeeze(1)  # (B, D_concat)

    # (1) 拆分各模态: (B, D_concat) → list of (B, D_per)
    modality_feats = features.split(self.d_per, dim=-1)

    # (2) 各模态独立投影 → (B, n_modalities, hidden_dim)
    projected = torch.stack([
      proj(feat) for proj, feat in zip(self.modality_projectors, modality_feats)
    ], dim=1)

    # (3) Gate 计算: concat 所有投影 → softmax 门控权重
    gate_input = projected.reshape(projected.size(0), -1)  # (B, n_mod * hidden)
    gate_logits = self.gate_net(gate_input)                 # (B, n_mod)
    gates = F.softmax(gate_logits / self.gate_temperature, dim=-1)

    # (4) 加权融合: (B, n_mod, 1) * (B, n_mod, hidden) → sum → (B, hidden)
    fused = (gates.unsqueeze(-1) * projected).sum(dim=1)

    # (5) 分类
    logits = self.classifier(fused)
    return {'logits': logits, 'gates': gates}
