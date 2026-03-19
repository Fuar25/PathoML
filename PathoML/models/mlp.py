"""MLP 单隐藏层分类模型。"""

from typing import Dict

import torch
import torch.nn as nn

from ..interfaces import BaseModel, DataDict
from ..registry import register_model


@register_model('mlp')
class MLP(BaseModel):
  """带单隐藏层和 Dropout 的 MLP 分类模型。

  适用于预计算 WSI 级别特征的分类，相比 LinearProbe 增加一层非线性变换，
  可作为轻量级基线与 ABMIL 等注意力模型对比。
  """

  def __init__(
    self,
    input_dim: int,
    hidden_dim: int,
    num_classes: int = 1,
    dropout: float = 0.0,
    **kwargs,
  ) -> None:
    """初始化 MLP 模型。

    Args:
      input_dim: 输入特征的维度。
      hidden_dim: 隐藏层维度。
      num_classes: 输出类别数（二分类时为 1）。
      dropout: Dropout 比率，应用于隐藏层激活后。为 0 时不使用 Dropout。
      **kwargs: 忽略的额外参数，保持与其他模型签名的兼容性。
    """
    super(MLP, self).__init__()

    self.net = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
      nn.Linear(hidden_dim, num_classes),
    )

  def forward(self, data_dict: DataDict) -> Dict[str, torch.Tensor]:
    """对输入特征执行单隐藏层 MLP 分类，返回 logits。

    支持形状为 (B, C) 的二维输入，也支持 Bag 维度为 1 的
    三维输入 (B, 1, C)，后者会自动压缩为二维。

    Args:
      data_dict: 输入字典，需包含 'features' 键：
        - 形状为 (B, C) 或 (B, 1, C)。

    Returns:
      包含 'logits' 键的输出字典，logits 形状为 (B, num_classes)。
    """
    features = data_dict['features']
    # 若 Bag 维度为 1（单实例），则压缩为二维
    if features.dim() == 3 and features.size(1) == 1:
      features = features.squeeze(1)
    logits = self.net(features)
    return {'logits': logits}
