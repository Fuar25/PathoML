"""DistillationDataset: 每个WSI样本加载HE patch H5文件及可配置的多个slide H5文件。

每个样本对应：
  - HE patch H5:          features (N, D_patch)
  - 若干 slide H5 文件:  features (D_slide,)，按 slide_stains 顺序 cat 为 slide_concat

文件名格式: <patient_id><tissue_id>-<stain>.h5
  示例: B2022-01475B-he.h5 → patient_id=B2022-01475, tissue_id=B

目录结构 (patient-based):
  <root>/<patient_id>/<tissue_id>/<patient_id><tissue_id>-<stain>.h5
  labels_csv  ← patient_id,label

类别标签: 按CSV的label列逆字母序排序。Reactive=0，MALT=1。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from PathoML.config.defaults import PATIENT_ID_PATTERN
from PathoML.dataset.utils import (
  _extract_patient_tissue_id, _walk_h5_files,
  find_common_sample_keys, load_labels_csv,
)
from PathoML.interfaces import BaseDataset


def _load_h5_features(path: str) -> torch.Tensor:
  """读取H5文件中的features数组，返回float32 Tensor。"""
  with h5py.File(path, 'r') as f:
    if 'features' not in f:
      raise KeyError(f"'features' key not found in H5 file: {path}")
    return torch.from_numpy(np.array(f['features'])).float()


def _build_key_map(
  root: str,
  pattern: str,
  label_map: Dict[str, str],
  stain: Optional[str] = None,
  allowed_keys: Optional[set] = None,
) -> Dict[Tuple[str, str], Tuple[str, str]]:
  """扫描root下的H5文件，返回 (patient_id, tissue_id) → (class_name, abs_path) 映射。

  Args:
    root: 数据根目录（递归扫描）。
    pattern: 患者ID正则表达式。
    label_map: patient_id → class_name 映射。
    stain: 可选染色过滤。
    allowed_keys: 可选白名单，仅保留命中的 (patient_id, tissue_id)。

  Returns:
    dict，键为(patient_id, tissue_id)，值为(class_name, abs_path)。
  """
  key_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
  if not os.path.isdir(root):
    raise FileNotFoundError(f"数据根目录不存在: {root}")
  for fname, filepath in _walk_h5_files(root, stain=stain):
    key = _extract_patient_tissue_id(fname, pattern)
    if key is None:
      continue
    if allowed_keys is not None and key not in allowed_keys:
      continue
    patient_id, _ = key
    class_name = label_map.get(patient_id)
    if class_name is None:
      continue
    key_map[key] = (class_name, filepath)
  return key_map


class DistillationDataset(BaseDataset):
  """蒸馏训练数据集：每样本同时提供HE patch特征和可配置模态的slide embedding拼接。

  用法:
    dataset = DistillationDataset(
      patch_root='/data/Patch',
      slide_root='/data/Slide',
      slide_stains=['HE', 'CD20', 'CD3'],
      labels_csv='labels.csv',
    )
    item = dataset[0]
    # item['he_patches']:   (N, D_patch)
    # item['slide_concat']: (D_he + D_cd20 + D_cd3,)  ← 按 slide_stains 顺序拼接
    # item['label']:        float tensor（0.0 或 1.0，BCE用）
  """

  def __init__(
    self,
    patch_root: str,
    slide_root: str,
    slide_stains: List[str],
    labels_csv: str,
    patch_stain: str = 'HE',
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    allowed_sample_keys: Optional[set] = None,
  ) -> None:
    """
    Args:
      patch_root:   HE patch H5根目录（features形状为 N×D）。
      slide_root:   Slide embedding H5根目录。
      slide_stains: Slide模态染色列表，决定 slide_concat 的拼接顺序。
      labels_csv:   CSV标签文件路径（patient_id,label）。
      patch_stain:  Patch模态的染色名（默认 HE）。
      patient_id_pattern: 患者ID正则，须与文件命名一致。
      allowed_sample_keys: 可选 (patient_id, tissue_id) 白名单。
                    传入时直接使用；不传时内部自动计算交集。
    """
    super().__init__()
    if not slide_stains:
      raise ValueError("slide_stains 不能为空。")
    self.patient_id_pattern = patient_id_pattern
    self.slide_stains = slide_stains
    self.patch_stain = patch_stain
    label_map = load_labels_csv(labels_csv)

    # (1) 取所有模态的公共样本键
    if allowed_sample_keys is not None:
      common_keys = allowed_sample_keys
    else:
      # Patch root + slide root 共用同一个 stain 列表计算交集
      all_stains = [patch_stain] + slide_stains
      # 分别在 patch_root 和 slide_root 下找公共键，再取交集
      patch_keys = find_common_sample_keys(patch_root, [patch_stain], patient_id_pattern)
      slide_keys = find_common_sample_keys(slide_root, slide_stains, patient_id_pattern)
      common_keys = patch_keys & slide_keys
      if not common_keys:
        raise FileNotFoundError(
          f"各根目录无公共样本。请检查路径与文件命名是否一致。"
        )

    # (2) 扫描各根目录，仅保留 common_keys 中的样本
    patch_map = _build_key_map(
      patch_root, patient_id_pattern, label_map,
      stain=patch_stain, allowed_keys=common_keys,
    )
    slide_maps: Dict[str, Dict] = {
      stain: _build_key_map(
        slide_root, patient_id_pattern, label_map,
        stain=stain, allowed_keys=common_keys,
      )
      for stain in slide_stains
    }

    # (3) 确定类别标签
    all_classes = sorted(set(label_map.values()), reverse=True)
    self.classes = all_classes
    self.class_to_label = {cls: i for i, cls in enumerate(all_classes)}

    # (4) 构建样本列表（按key排序，保证reproducibility）
    self.samples: List[dict] = []
    for key in sorted(common_keys):
      if key not in patch_map:
        continue
      patient_id, tissue_id = key
      class_name, patch_path = patch_map[key]
      if any(key not in slide_maps[stain] for stain in self.slide_stains):
        continue
      slide_paths = {stain: slide_maps[stain][key][1] for stain in self.slide_stains}
      slide_id = os.path.splitext(os.path.basename(patch_path))[0]
      self.samples.append({
        'patient_id':   patient_id,
        'tissue_id':    tissue_id,
        'slide_id':     slide_id,
        'label':        self.class_to_label[class_name],
        'patch_path':   patch_path,
        'slide_paths':  slide_paths,
      })

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> dict:
    s = self.samples[idx]
    he_patches = _load_h5_features(s['patch_path'])   # (N, D)
    # (1) 加载各模态 slide embedding 并拼接
    slide_tensors = [
      _load_h5_features(s['slide_paths'][stain]).view(-1)
      for stain in self.slide_stains
    ]
    slide_concat = torch.cat(slide_tensors, dim=0)     # (sum_of_dims,)
    return {
      'he_patches':   he_patches,
      'slide_concat': slide_concat,
      'label':        torch.tensor(s['label'], dtype=torch.float32),
      'patient_id':   s['patient_id'],
      'tissue_id':    s['tissue_id'],
      'slide_id':     s['slide_id'],
    }

  def get_patient_ids(self) -> List[str]:
    """返回所有样本的patient_id列表（与self.samples顺序一致）。"""
    return [s['patient_id'] for s in self.samples]

  def get_labels(self) -> List[int]:
    """返回所有样本的标签列表（与self.samples顺序一致）。"""
    return [s['label'] for s in self.samples]
