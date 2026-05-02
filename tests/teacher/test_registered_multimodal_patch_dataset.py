"""Tests for registered multimodal patch datasets."""

from __future__ import annotations

import h5py
import numpy as np

from teacher.dataset import RegisteredMultimodalPatchDataset


def _write_h5(path, features, coords) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with h5py.File(path, 'w') as f:
    f.create_dataset('features', data=np.asarray(features, dtype=np.float32))
    f.create_dataset('coords', data=np.asarray(coords, dtype=np.int64))


def test_registered_multimodal_patch_dataset_aligns_by_shared_coords(tmp_path):
  labels_csv = tmp_path / 'labels.csv'
  labels_csv.write_text('patient_id,label\nB2020-00001,MALT\n', encoding='utf-8')
  tissue_dir = tmp_path / 'B2020-00001' / 'A'

  _write_h5(
    tissue_dir / 'B2020-00001A-HE.h5',
    features=[[1, 10], [2, 20], [3, 30]],
    coords=[[0, 0], [1, 1], [2, 2]],
  )
  _write_h5(
    tissue_dir / 'B2020-00001A-cd20.h5',
    features=[[200, 2000], [100, 1000], [400, 4000]],
    coords=[[2, 2], [1, 1], [4, 4]],
  )
  _write_h5(
    tissue_dir / 'B2020-00001A-cd3.h5',
    features=[[300, 3000], [100, 1000], [500, 5000]],
    coords=[[2, 2], [1, 1], [5, 5]],
  )

  dataset = RegisteredMultimodalPatchDataset(
    data_root=str(tmp_path),
    modality_names=['HE', 'CD20', 'CD3'],
    labels_csv=str(labels_csv),
  )

  assert len(dataset) == 1
  item = dataset[0]
  assert item['aligned_patch_count'] == 2
  np.testing.assert_array_equal(item['coords'].numpy(), [[1, 1], [2, 2]])
  np.testing.assert_array_equal(
    item['features'].numpy(),
    [
      [2, 20, 100, 1000, 100, 1000],
      [3, 30, 200, 2000, 300, 3000],
    ],
  )


def test_registered_multimodal_patch_dataset_filters_low_alignment(tmp_path):
  labels_csv = tmp_path / 'labels.csv'
  labels_csv.write_text('patient_id,label\nB2020-00001,MALT\n', encoding='utf-8')
  tissue_dir = tmp_path / 'B2020-00001' / 'A'

  _write_h5(tissue_dir / 'B2020-00001A-HE.h5', [[1]], [[0, 0]])
  _write_h5(tissue_dir / 'B2020-00001A-cd20.h5', [[2]], [[1, 1]])
  _write_h5(tissue_dir / 'B2020-00001A-cd3.h5', [[3]], [[2, 2]])

  dataset = RegisteredMultimodalPatchDataset(
    data_root=str(tmp_path),
    modality_names=['HE', 'CD20', 'CD3'],
    labels_csv=str(labels_csv),
    min_aligned_patches=1,
  )

  assert len(dataset) == 0


def test_registered_multimodal_patch_dataset_union_keeps_missing_modalities(tmp_path):
  labels_csv = tmp_path / 'labels.csv'
  labels_csv.write_text('patient_id,label\nB2020-00001,MALT\n', encoding='utf-8')
  tissue_dir = tmp_path / 'B2020-00001' / 'A'

  _write_h5(
    tissue_dir / 'B2020-00001A-HE.h5',
    features=[[1, 10], [2, 20]],
    coords=[[0, 0], [1, 1]],
  )
  _write_h5(
    tissue_dir / 'B2020-00001A-cd20.h5',
    features=[[3, 30]],
    coords=[[1, 1]],
  )
  _write_h5(
    tissue_dir / 'B2020-00001A-cd3.h5',
    features=[[4, 40]],
    coords=[[2, 2]],
  )

  dataset = RegisteredMultimodalPatchDataset(
    data_root=str(tmp_path),
    modality_names=['HE', 'CD20', 'CD3'],
    labels_csv=str(labels_csv),
    alignment_mode='union',
  )

  item = dataset[0]

  assert item['aligned_patch_count'] == 3
  np.testing.assert_array_equal(item['coords'].numpy(), [[0, 0], [1, 1], [2, 2]])
  np.testing.assert_array_equal(
    item['modality_mask'].numpy(),
    [
      [True, False, False],
      [True, True, False],
      [False, False, True],
    ],
  )
  np.testing.assert_array_equal(
    item['features'].numpy(),
    [
      [1, 10, 0, 0, 0, 0],
      [2, 20, 3, 30, 0, 0],
      [0, 0, 0, 0, 4, 40],
    ],
  )
