"""Linear Probe 线性探针基线模型。"""

from typing import Dict

import torch
import torch.nn as nn

from ..optimization.interfaces import BaseModel, DataDict
from ..optimization.registry import register_model


@register_model('linear_probe')
class LinearProbe(BaseModel):
    """基于预计算 WSI 级别特征的线性探针分类模型。

    直接对 WSI 级别的聚合特征（如均值池化后的 Bag 嵌入）
    应用单层线性分类器，适用于快速基线验证。
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        dropout: float = 0,
        **kwargs,
    ) -> None:
        """初始化线性探针模型。

        Args:
            input_dim: 输入特征的维度。
            num_classes: 输出类别数（二分类时为 1）。
            dropout: Dropout 比率。为 0 时不使用 Dropout。
            **kwargs: 忽略的额外参数，保持与 ABMIL 签名的兼容性。
        """
        super(LinearProbe, self).__init__()

        self.linear = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, data_dict: DataDict) -> Dict[str, torch.Tensor]:
        """对输入特征执行 Dropout 和线性分类，返回 logits。

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
        features = self.dropout(features)
        logits = self.linear(features)
        return {'logits': logits}

