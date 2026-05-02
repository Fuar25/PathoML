"""Teacher dataset: registered multimodal patch features."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from PathoML.config.defaults import PATIENT_ID_PATTERN
from PathoML.dataset.base import MultimodalSlideDatasetBase
from PathoML.registry import register_dataset


@register_dataset('RegisteredMultimodalPatchDataset')
class RegisteredMultimodalPatchDataset(MultimodalSlideDatasetBase):
  """Patch-level multimodal dataset aligned by shared registered coordinates."""

  def __init__(
    self,
    data_root: str,
    modality_names: List[str],
    labels_csv: str,
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    min_aligned_patches: int = 1,
    alignment_mode: str = 'inner',
    cache_aligned: bool = False,
    verbose: bool = True,
    allowed_sample_keys: Optional[Set[Tuple[str, str]]] = None,
  ) -> None:
    self.min_aligned_patches = int(min_aligned_patches)
    self.alignment_mode = alignment_mode
    self.cache_aligned = bool(cache_aligned)
    self._aligned_cache: list[Dict[str, Any]] = []
    if self.alignment_mode not in {'inner', 'union'}:
      raise ValueError("alignment_mode must be 'inner' or 'union'")
    super().__init__(
      data_root=data_root,
      modality_names=modality_names,
      labels_csv=labels_csv,
      patient_id_pattern=patient_id_pattern,
      allow_missing_modalities=False,
      verbose=False,
      allowed_sample_keys=allowed_sample_keys,
    )
    self._filter_by_aligned_patch_count()
    if verbose:
      print(
        f"RegisteredMultimodalPatchDataset: {len(self.samples)} samples, "
        f"modalities={self.modality_names}, min_aligned_patches={self.min_aligned_patches}, "
        f"alignment_mode={self.alignment_mode}, cache_aligned={self.cache_aligned}"
      )

  def _filter_by_aligned_patch_count(self) -> None:
    if self.min_aligned_patches <= 0 and not self.cache_aligned:
      return
    kept = []
    cached_items = []
    for item in self.samples:
      sample_key = item['sample_key']
      loaded_features, loaded_coords = self._load_modality_features(sample_key)
      aligned_coords = self._aligned_coords(loaded_features, loaded_coords)
      if len(aligned_coords) >= self.min_aligned_patches:
        item['aligned_patch_count'] = len(aligned_coords)
        kept.append(item)
        if self.cache_aligned:
          cached_items.append(
            self._build_item(item, loaded_features, loaded_coords, aligned_coords)
          )
    self.samples = kept
    self._aligned_cache = cached_items

  def _aligned_coords(
    self,
    loaded_features: Dict[str, torch.Tensor],
    loaded_coords: Dict[str, torch.Tensor],
  ) -> list[tuple[int, int]]:
    for modality_name in self.modality_names:
      if modality_name not in loaded_features:
        raise RuntimeError(f"Missing modality '{modality_name}' for registered sample")
      if modality_name not in loaded_coords:
        raise RuntimeError(f"Missing coords for modality '{modality_name}'")

    coord_sets = [
      {tuple(int(v) for v in coord.tolist()) for coord in loaded_coords[modality_name]}
      for modality_name in self.modality_names
    ]
    selected = (
      set.intersection(*coord_sets)
      if self.alignment_mode == 'inner'
      else set.union(*coord_sets)
    )

    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for modality_name in self.modality_names:
      for coord in loaded_coords[modality_name]:
        key = tuple(int(v) for v in coord.tolist())
        if key in selected and key not in seen:
          ordered.append(key)
          seen.add(key)
    return ordered

  @staticmethod
  def _coord_index(coords: torch.Tensor) -> dict[tuple[int, int], int]:
    index: dict[tuple[int, int], int] = {}
    for idx, coord in enumerate(coords):
      key = tuple(int(v) for v in coord.tolist())
      index.setdefault(key, idx)
    return index

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    if self.cache_aligned:
      return dict(self._aligned_cache[idx])

    item = self.samples[idx]
    sample_key = item['sample_key']
    loaded_features, loaded_coords = self._load_modality_features(sample_key)
    aligned_coords = self._aligned_coords(loaded_features, loaded_coords)
    if len(aligned_coords) < self.min_aligned_patches:
      raise RuntimeError(
        f"Sample {sample_key} has {len(aligned_coords)} aligned patches, "
        f"below min_aligned_patches={self.min_aligned_patches}"
      )
    return self._build_item(item, loaded_features, loaded_coords, aligned_coords)

  def _build_item(
    self,
    item: Dict[str, Any],
    loaded_features: Dict[str, torch.Tensor],
    loaded_coords: Dict[str, torch.Tensor],
    aligned_coords: list[tuple[int, int]],
  ) -> Dict[str, Any]:
    sample_key = item['sample_key']
    concat_parts: list[torch.Tensor] = []
    modality_masks: list[torch.Tensor] = []
    for modality_name in self.modality_names:
      coord_to_idx = self._coord_index(loaded_coords[modality_name])
      feature_dim = loaded_features[modality_name].shape[-1]
      parts = []
      present = []
      for coord in aligned_coords:
        idx_for_coord = coord_to_idx.get(coord)
        if idx_for_coord is None:
          parts.append(loaded_features[modality_name].new_zeros(feature_dim))
          present.append(False)
        else:
          parts.append(loaded_features[modality_name][idx_for_coord])
          present.append(True)
      concat_parts.append(torch.stack(parts, dim=0))
      modality_masks.append(torch.tensor(present, dtype=torch.bool))

    features = torch.cat(concat_parts, dim=-1)
    modality_mask = torch.stack(modality_masks, dim=-1)
    coords = torch.tensor(aligned_coords, dtype=torch.float32)
    label_tensor = (
      torch.tensor(item['label']).float()
      if self.binary_mode
      else torch.tensor(item['label']).long()
    )
    return {
      'features': features,
      'coords': coords,
      'modality_mask': modality_mask,
      'label': label_tensor,
      'slide_id': f"{sample_key[0]}{sample_key[1]}",
      'patient_id': item['patient_id'],
      'tissue_id': item['tissue_id'],
      'modalities': list(self.modality_names),
      'aligned_patch_count': len(aligned_coords),
    }

  def get_item_length(self, idx: int) -> int:
    if self.cache_aligned:
      return int(self._aligned_cache[idx]['aligned_patch_count'])
    item = self.samples[idx]
    if 'aligned_patch_count' in item:
      return int(item['aligned_patch_count'])
    sample_key = item['sample_key']
    loaded_features, loaded_coords = self._load_modality_features(sample_key)
    aligned_coords = self._aligned_coords(loaded_features, loaded_coords)
    item['aligned_patch_count'] = len(aligned_coords)
    return len(aligned_coords)
