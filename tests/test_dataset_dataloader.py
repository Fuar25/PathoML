"""Integration tests: SyntheticDataset + PyTorch DataLoader."""

import torch
from torch.utils.data import DataLoader

from PathoML.dataset.utils import _variable_size_collate


def test_dataset_len_and_item_shapes(synthetic_dataset):
  assert len(synthetic_dataset) == 40

  item = synthetic_dataset[0]
  assert item['features'].shape == (5, 32)
  assert item['label'].ndim == 0              # 0-dim scalar tensor
  assert item['label'].dtype == torch.float32
  assert isinstance(item['sample_id'], str)
  assert isinstance(item['patient_id'], str)


def test_dataloader_batch_shape(synthetic_dataset):
  loader = DataLoader(synthetic_dataset, batch_size=1, shuffle=False)
  batch = next(iter(loader))

  # (1) DataLoader adds a batch dimension: (N, C) → (1, N, C)
  assert batch['features'].shape == (1, 5, 32)
  assert batch['label'].shape == (1,)


# ---------------------------------------------------------------------------
# _variable_size_collate tests
# ---------------------------------------------------------------------------

def test_collate_same_shape_no_mask():
  """When all tensors have the same shape, stack normally — no mask created."""
  batch = [
    {'features': torch.randn(5, 8), 'label': torch.tensor(0.0), 'id': 'a'},
    {'features': torch.randn(5, 8), 'label': torch.tensor(1.0), 'id': 'b'},
  ]
  result = _variable_size_collate(batch)
  assert result['features'].shape == (2, 5, 8)
  assert result['label'].shape == (2,)
  assert 'mask' not in result
  assert result['id'] == ['a', 'b']


def test_collate_variable_n_pads_and_masks():
  """Variable N along dim 0 → pad to max_N, create mask."""
  batch = [
    {'features': torch.ones(3, 4), 'coords': torch.ones(3, 2), 'label': torch.tensor(0.0)},
    {'features': torch.ones(5, 4), 'coords': torch.ones(5, 2), 'label': torch.tensor(1.0)},
  ]
  result = _variable_size_collate(batch)

  # (1) Features and coords padded to max_N=5
  assert result['features'].shape == (2, 5, 4)
  assert result['coords'].shape == (2, 5, 2)

  # (2) Mask shape and values
  assert result['mask'].shape == (2, 5)
  assert result['mask'][0].tolist() == [True, True, True, False, False]
  assert result['mask'][1].tolist() == [True, True, True, True, True]

  # (3) Padding positions are zero
  assert (result['features'][0, 3:] == 0).all()
  assert (result['coords'][0, 3:] == 0).all()

  # (4) Original data preserved
  assert (result['features'][0, :3] == 1).all()
  assert (result['features'][1] == 1).all()


def test_collate_labels_stacked():
  """Scalar labels are stacked into a 1-D tensor regardless of N variation."""
  batch = [
    {'features': torch.randn(3, 4), 'label': torch.tensor(0.0)},
    {'features': torch.randn(7, 4), 'label': torch.tensor(1.0)},
  ]
  result = _variable_size_collate(batch)
  assert result['label'].shape == (2,)
  assert result['label'].tolist() == [0.0, 1.0]
