"""Teacher dataset: slide-level unimodal WSI features."""

from __future__ import annotations

from PathoML.dataset.base import UnimodalFeatureDatasetBase
from PathoML.registry import register_dataset


@register_dataset('UnimodalSlideDataset')
class UnimodalSlideDataset(UnimodalFeatureDatasetBase):
  """Slide-level unimodal dataset for teacher experiments."""

