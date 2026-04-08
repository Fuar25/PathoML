"""Teacher dataset: patch-level unimodal WSI features."""

from __future__ import annotations

from PathoML.dataset.base import UnimodalFeatureDatasetBase
from PathoML.registry import register_dataset


@register_dataset('UnimodalPatchDataset')
class UnimodalPatchDataset(UnimodalFeatureDatasetBase):
  """Patch-level unimodal dataset for teacher experiments."""

