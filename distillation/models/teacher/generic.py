"""Generic registry-backed teacher adapter for distillation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from PathoML.registry import create_model
from teacher.runtime.loader import load_teacher_modules


class RegistryTeacher(nn.Module):
  """Load a teacher registry model and normalize outputs for distillation losses."""

  def __init__(self, model: nn.Module) -> None:
    super().__init__()
    self.model = model
    self.train_fold = None
    self.test_fold = None

  @classmethod
  def from_manifest_checkpoint(
    cls,
    manifest: Any,
    ckpt_path: str,
    dataset: Any,
  ) -> "RegistryTeacher":
    """Build a frozen teacher model from a manifest and checkpoint."""
    load_teacher_modules()
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(raw, dict) and 'state_dict' in raw:
      state = raw['state_dict']
      train_fold = raw.get('train_fold', None)
      test_fold = raw.get('test_fold', None)
    else:
      state = raw
      train_fold = test_fold = None

    sample = dataset[0]
    input_dim = int(sample['features'].shape[-1])
    n_classes = len(dataset.classes)
    num_classes = 1 if n_classes == 2 else n_classes
    model = create_model(
      manifest.model_name,
      input_dim=input_dim,
      num_classes=num_classes,
      **manifest.model_kwargs,
    )
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
      param.requires_grad = False

    adapter = cls(model)
    adapter.train_fold = train_fold
    adapter.test_fold = test_fold
    return adapter

  def _class_weight(self) -> torch.Tensor | None:
    classifier = getattr(self.model, 'classifier', None)
    linear = getattr(classifier, 'linear', None)
    weight = getattr(linear, 'weight', None)
    if weight is None:
      return None
    return weight.squeeze(0)

  def forward(self, batch: dict) -> dict:
    out = self.model(batch)
    hidden = out.get('hidden')
    if hidden is None:
      hidden = out.get('bag_embeddings')
    logit = out.get('logit')
    if logit is None:
      logit = out.get('logits')
    if hidden is None or logit is None:
      raise ValueError(
        "Teacher forward must expose hidden/bag_embeddings and logit/logits."
      )

    normalized = {
      'hidden': hidden,
      'logit': logit,
    }
    if 'attention' in out:
      normalized['attention'] = out['attention']
    if 'attn_logits' in out:
      normalized['attn_logits'] = out['attn_logits']
    class_weight = out.get('class_weight')
    if class_weight is None:
      class_weight = self._class_weight()
    if class_weight is not None:
      normalized['class_weight'] = class_weight
    return normalized
