"""Base distillation loss contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class DistillationLoss(nn.Module, ABC):
  """Stable extension-point contract for distillation losses."""

  @abstractmethod
  def forward(
    self,
    s_out: dict,
    t_out: dict,
    labels: Tensor,
  ) -> Tensor:
    ...
