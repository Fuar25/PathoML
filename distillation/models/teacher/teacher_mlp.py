"""TeacherMLP: 加载已训练的PathoML MLP checkpoint，暴露256-dim hidden用于蒸馏。

与PathoML MLP结构完全一致（Linear+GELU+Dropout+Linear），
可直接加载PathoML MLP保存的state_dict（键名：net.0.*, net.3.*）。

用法:
  teacher = TeacherMLP.from_checkpoint('path/to/teacher_fold_k.pth')
  out = teacher(x)  # x: (B, 1536)，input_dim 自动从 checkpoint 推断
  # out['hidden']: (B, 256)，Linear+GELU后（无Dropout）
  # out['logit']:  (B, 1)，最终BCE logit
  # out['class_weight']: (hidden_dim,)，最终分类器的正类方向
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TeacherMLP(nn.Module):
  """BCE binary MLP，与PathoML MLP checkpoint完全兼容，额外暴露hidden层。"""

  def __init__(
    self,
    input_dim: int = 1536,
    hidden_dim: int = 256,
    dropout: float = 0.3,
  ) -> None:
    """
    Args:
      input_dim: concat(HE, CD20)的维度，默认1536（768+768）。
      hidden_dim: 隐藏层维度，默认256。
      dropout: Dropout比率，须与保存checkpoint时一致。
    """
    super().__init__()
    # (1) 与PathoML MLP完全一致的net结构，确保state_dict键名对齐
    self.net = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),               # net.0
      nn.GELU(),                                       # net.1
      nn.Dropout(dropout) if dropout > 0 else nn.Identity(),  # net.2
      nn.Linear(hidden_dim, 1),                        # net.3
    )

  def forward(self, x: torch.Tensor) -> dict:
    """
    Args:
      x: concat后的slide embedding，形状 (B, input_dim)。

    Returns:
      dict，包含：
        - 'hidden': (B, hidden_dim)，Linear+GELU后、Dropout前的表示
        - 'logit':  (B, 1)，用于BCE的原始logit
        - 'class_weight': (hidden_dim,)，最终分类器正类方向
    """
    hidden = self.net[:2](x)       # Linear + GELU，不含Dropout
    logit = self.net[2:](hidden)   # Dropout + Linear
    class_weight = self.net[3].weight.squeeze(0)
    return {'hidden': hidden, 'logit': logit, 'class_weight': class_weight}

  @classmethod
  def from_checkpoint(cls, ckpt_path: str) -> 'TeacherMLP':
    """从 PathoML MLP checkpoint 自动推断维度并加载，无需手动指定参数。

    支持新格式（含 fold 元数据的 dict）和旧格式（裸 state_dict）。
    fold 元数据通过实例属性 train_fold / test_fold 暴露（旧格式为 None）。

    Args:
      ckpt_path: PathoML MLP保存的checkpoint路径（.pth）。

    Returns:
      参数完全冻结、eval模式的TeacherMLP实例。
    """
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    # (1) 解包新格式（含元数据）或旧格式（裸 state_dict）
    if isinstance(raw, dict) and 'state_dict' in raw:
      state      = raw['state_dict']
      train_fold = raw.get('train_fold', None)
      test_fold  = raw.get('test_fold',  None)
    else:
      state      = raw    # 旧格式兼容
      train_fold = test_fold = None
    # (2) 从 state_dict 推断维度（net.0 为第一层 Linear）
    input_dim  = state['net.0.weight'].shape[1]
    hidden_dim = state['net.0.weight'].shape[0]
    # eval模式下 Dropout 等价于 Identity，dropout值不影响推理结果
    model = cls(input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.0)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
      p.requires_grad = False
    # (3) 暴露 fold 元数据供蒸馏时校验
    model.train_fold = train_fold
    model.test_fold  = test_fold
    return model

  @classmethod
  def load_frozen(cls, ckpt_path: str, **kwargs) -> 'TeacherMLP':
    """加载checkpoint并冻结所有参数（须手动指定维度）。

    推荐使用 from_checkpoint() 替代，可自动推断维度。

    Args:
      ckpt_path: PathoML MLP保存的state_dict路径（.pth）。
      **kwargs: 传给__init__（input_dim, hidden_dim, dropout）。
    """
    model = cls(**kwargs)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
      p.requires_grad = False
    return model
