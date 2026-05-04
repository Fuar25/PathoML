"""Tests for distillation in-memory caches."""

from __future__ import annotations

import csv

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PathoML.config.config import RunTimeConfig
from distillation.dataset import DistillationDataset
from distillation.losses import DistillationLoss
from distillation.runtime import DistillCrossValidator


def _write_h5(path, values):
  path.parent.mkdir(parents=True, exist_ok=True)
  with h5py.File(path, 'w') as f:
    f.create_dataset('features', data=np.asarray(values, dtype=np.float32))


def _write_labels(path, patient_ids):
  with path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['patient_id', 'label'])
    writer.writeheader()
    for idx, patient_id in enumerate(patient_ids):
      writer.writerow({
        'patient_id': patient_id,
        'label': 'MALT' if idx % 2 == 0 else 'Reactive',
      })


def test_distillation_dataset_cache_matches_uncached_items(tmp_path):
  patch_root = tmp_path / 'patch'
  slide_root = tmp_path / 'slide'
  labels_csv = tmp_path / 'labels.csv'
  sample_ids = [('B2020-00001', 'A'), ('B2020-00002', 'B')]
  _write_labels(labels_csv, [patient_id for patient_id, _ in sample_ids])

  for sample_idx, (patient_id, tissue_id) in enumerate(sample_ids):
    prefix = f'{patient_id}{tissue_id}'
    _write_h5(
      patch_root / patient_id / tissue_id / f'{prefix}-HE.h5',
      np.full((sample_idx + 2, 3), sample_idx + 1),
    )
    for stain_idx, stain in enumerate(['HE', 'CD20']):
      _write_h5(
        slide_root / patient_id / tissue_id / f'{prefix}-{stain}.h5',
        np.full((1, 2), sample_idx * 10 + stain_idx),
      )

  cached = DistillationDataset(
    patch_root=str(patch_root),
    slide_root=str(slide_root),
    slide_stains=['HE', 'CD20'],
    labels_csv=str(labels_csv),
    cache_features=True,
  )
  uncached = DistillationDataset(
    patch_root=str(patch_root),
    slide_root=str(slide_root),
    slide_stains=['HE', 'CD20'],
    labels_csv=str(labels_csv),
    cache_features=False,
  )

  assert len(cached) == len(uncached) == 2
  for idx in range(len(cached)):
    cached_item = cached[idx]
    uncached_item = uncached[idx]
    assert cached_item['slide_id'] == uncached_item['slide_id']
    assert cached_item['label'].item() == uncached_item['label'].item()
    assert cached_item['sample_index'].item() == idx
    torch.testing.assert_close(cached_item['he_patches'], uncached_item['he_patches'])
    torch.testing.assert_close(cached_item['slide_concat'], uncached_item['slide_concat'])
    torch.testing.assert_close(cached.get_slide_concat(idx), uncached.get_slide_concat(idx))


class _TinyDataset:
  classes = ['Reactive', 'MALT']

  def __init__(self):
    self.slide = torch.tensor([
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0],
    ])
    self.labels = [0, 1, 0]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return {
      'he_patches': torch.ones(2, 2),
      'slide_concat': self.slide[idx],
      'label': torch.tensor(float(self.labels[idx])),
      'patient_id': f'P{idx}',
      'slide_id': f'S{idx}',
      'sample_index': torch.tensor(idx, dtype=torch.long),
    }

  def get_slide_concat(self, idx):
    return self.slide[idx]

  def get_patient_ids(self):
    return [f'P{idx}' for idx in range(len(self))]

  def get_labels(self):
    return list(self.labels)


class _CountingTeacher(nn.Module):
  def __init__(self):
    super().__init__()
    self.forward_calls = 0

  def forward(self, x):
    self.forward_calls += 1
    return {
      'hidden': x + 1.0,
      'logit': x.sum(dim=1, keepdim=True),
      'class_weight': torch.tensor([0.25, -0.25], dtype=x.dtype, device=x.device),
    }


class _TinyStudent(nn.Module):
  def __init__(self):
    super().__init__()
    self.bias = nn.Parameter(torch.tensor(0.5))

  def forward(self, data):
    batch_size = data['he_patches'].size(0)
    logits = self.bias.expand(batch_size, 1)
    return {'logits': logits}


class _CacheAssertingLoss(DistillationLoss):
  def forward(self, s_out, t_out, labels):
    assert 'hidden' in t_out
    assert 'logit' in t_out
    assert 'class_weight' in t_out
    return s_out['logits'].pow(2).mean() + t_out['hidden'].sum() * 0.0 + labels.sum() * 0.0

  def describe(self):
    return 'L_cache_assert'

  def slug(self):
    return 'cache_assert'


def _make_cv(cache_teacher_outputs=True):
  config = RunTimeConfig()
  config.training.device = 'cpu'
  config.training.batch_size = 2
  return DistillCrossValidator(
    student_builder=lambda: _TinyStudent(),
    dataset=_TinyDataset(),
    config=config,
    distill_loss=_CacheAssertingLoss(),
    teacher_ckpt_tmpl='unused',
    k_folds=2,
    cache_teacher_outputs=cache_teacher_outputs,
    teacher_output_cache_batch_size=2,
  )


def test_teacher_output_cache_matches_live_teacher_forward():
  cv = _make_cv()
  cv.teacher = _CountingTeacher()

  cache = cv._precompute_teacher_outputs()
  live = cv.teacher(cv.dataset.slide)

  torch.testing.assert_close(cache['hidden'], live['hidden'])
  torch.testing.assert_close(cache['logit'], live['logit'])
  torch.testing.assert_close(cache['class_weight'], live['class_weight'])


def test_train_epoch_uses_cached_teacher_outputs_without_live_forward():
  cv = _make_cv()
  cv.teacher = _CountingTeacher()
  cv.teacher_output_cache = cv._precompute_teacher_outputs()
  cv.teacher.forward_calls = 0
  loader = DataLoader(cv.dataset, batch_size=2, shuffle=False)
  model = _TinyStudent()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

  loss, _ = cv._train_epoch(model, loader, nn.BCEWithLogitsLoss(), optimizer)

  assert loss > 0.0
  assert cv.teacher.forward_calls == 0
