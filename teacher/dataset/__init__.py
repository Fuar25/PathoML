"""Teacher-specific concrete datasets."""

from .multimodal_concat_slide import MultimodalConcatSlideDataset
from .multimodal_fusion_slide import MultimodalFusionSlideDataset
from .registered_multimodal_patch import RegisteredMultimodalPatchDataset
from .unimodal_patch import UnimodalPatchDataset
from .unimodal_slide import UnimodalSlideDataset

__all__ = [
  'MultimodalConcatSlideDataset',
  'MultimodalFusionSlideDataset',
  'RegisteredMultimodalPatchDataset',
  'UnimodalPatchDataset',
  'UnimodalSlideDataset',
]
