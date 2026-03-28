"""DistillationDataset: 每个WSI样本加载HE patch H5文件及可配置的多个slide H5文件。

每个样本对应：
  - HE patch H5:          features (N, D_patch)
  - 若干 slide H5 文件:  features (D_slide,)，按 slide_roots 顺序 cat 为 slide_concat

文件名格式: <patient_id><tissue_id>-<stain>.h5
  示例: B2022-01475B-he.h5 → patient_id=B2022-01475, tissue_id=B

目录结构（所有根目录均需此结构）:
  <root>/
    MALT/
      *.h5
    Reactive/
      *.h5

类别标签: 子目录名按字母序排序，MALT=0，Reactive=1。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from PathoML.config.defaults import PATIENT_ID_PATTERN
from PathoML.dataset.utils import _extract_patient_tissue_id, find_common_sample_keys
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
  allowed_keys: Optional[set] = None,
) -> Dict[Tuple[str, str], Tuple[str, str]]:
  """扫描root下的H5文件，返回 (patient_id, tissue_id) → (class_name, abs_path) 映射。

  Args:
    root: 数据根目录，含MALT/和Reactive/等类别子目录。
    pattern: 患者ID正则表达式。
    allowed_keys: 可选白名单，仅保留命中的 (patient_id, tissue_id)。

  Returns:
    dict，键为(patient_id, tissue_id)，值为(class_name, abs_path)。
  """
  key_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
  if not os.path.isdir(root):
    raise FileNotFoundError(f"数据根目录不存在: {root}")
  for class_name in sorted(os.listdir(root)):
    class_dir = os.path.join(root, class_name)
    if not os.path.isdir(class_dir):
      continue
    for dirpath, _, filenames in os.walk(class_dir):
      for fname in filenames:
        if not fname.endswith('.h5'):
          continue
        key = _extract_patient_tissue_id(fname, pattern)
        if key is None:
          continue
        if allowed_keys is not None and key not in allowed_keys:
          continue
        key_map[key] = (class_name, os.path.join(dirpath, fname))
  return key_map


class DistillationDataset(BaseDataset):
  """蒸馏训练数据集：每样本同时提供HE patch特征和可配置模态的slide embedding拼接。

  用法:
    dataset = DistillationDataset(
      patch_root='/data/patch_he',
      slide_roots={
        'he':   '/data/slide_he',
        'cd20': '/data/slide_cd20',
        'cd3':  '/data/slide_cd3',
      },
    )
    item = dataset[0]
    # item['he_patches']:   (N, D_patch)
    # item['slide_concat']: (D_he + D_cd20 + D_cd3,)  ← 按 slide_roots 顺序拼接
    # item['label']:        float tensor（0.0 或 1.0，BCE用）
  """

  def __init__(
    self,
    patch_root: str,
    slide_roots: Dict[str, str],
    patient_id_pattern: str = PATIENT_ID_PATTERN,
    allowed_sample_keys: Optional[set] = None,
  ) -> None:
    """
    Args:
      patch_root:   HE patch H5根目录（features形状为 N×D）。
      slide_roots:  模态名 → slide embedding H5根目录的映射（features形状为 1×D 或 D）。
                    dict 插入顺序决定 slide_concat 的拼接顺序。
                    示例: {'he': '/data/slide_he', 'cd20': '/data/slide_cd20'}
      patient_id_pattern: 患者ID正则，须与文件命名一致。
      allowed_sample_keys: 可选 (patient_id, tissue_id) 白名单。
                    传入时直接使用；不传时内部调用 find_common_sample_keys 自动计算交集。
    """
    super().__init__()
    if not slide_roots:
      raise ValueError("slide_roots 不能为空。")
    self.patient_id_pattern = patient_id_pattern
    self.slide_modalities: List[str] = list(slide_roots.keys())

    # (1) 取所有模态的公共样本键
    if allowed_sample_keys is not None:
      common_keys = allowed_sample_keys
    else:
      all_dirs = [patch_root] + list(slide_roots.values())
      common_keys = find_common_sample_keys(all_dirs, patient_id_pattern)
      if not common_keys:
        raise FileNotFoundError(
          f"各根目录无公共样本。请检查路径与文件命名是否一致。"
        )

    # (2) 扫描各根目录，仅保留 common_keys 中的样本
    patch_map = _build_key_map(patch_root, patient_id_pattern, common_keys)
    slide_maps: Dict[str, Dict] = {
      stain: _build_key_map(root, patient_id_pattern, common_keys)
      for stain, root in slide_roots.items()
    }

    # (3) 确定类别标签（字母序，MALT=0, Reactive=1）
    all_classes = sorted({v[0] for v in patch_map.values()})
    self.classes = all_classes
    self.class_to_label = {cls: i for i, cls in enumerate(all_classes)}

    # (4) 构建样本列表（按key排序，保证reproducibility）
    self.samples: List[dict] = []
    for key in sorted(common_keys):
      if key not in patch_map:
        continue
      patient_id, tissue_id = key
      class_name, patch_path = patch_map[key]
      if any(key not in slide_maps[stain] for stain in self.slide_modalities):
        continue
      slide_paths = {stain: slide_maps[stain][key][1] for stain in self.slide_modalities}
      sample_id = os.path.splitext(os.path.basename(patch_path))[0]
      self.samples.append({
        'patient_id':   patient_id,
        'tissue_id':    tissue_id,
        'sample_id':    sample_id,
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
      for stain in self.slide_modalities
    ]
    slide_concat = torch.cat(slide_tensors, dim=0)     # (sum_of_dims,)
    return {
      'he_patches':   he_patches,
      'slide_concat': slide_concat,
      'label':        torch.tensor(s['label'], dtype=torch.float32),
      'patient_id':   s['patient_id'],
      'tissue_id':    s['tissue_id'],
      'sample_id':    s['sample_id'],
    }

  def get_patient_ids(self) -> List[str]:
    """返回所有样本的patient_id列表（与self.samples顺序一致）。"""
    return [s['patient_id'] for s in self.samples]

  def get_labels(self) -> List[int]:
    """返回所有样本的标签列表（与self.samples顺序一致）。"""
    return [s['label'] for s in self.samples]
