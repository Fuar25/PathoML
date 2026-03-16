"""Integration tests: SyntheticDataset + PyTorch DataLoader."""

import torch
from torch.utils.data import DataLoader


def test_dataset_len_and_item_shapes(synthetic_dataset):
  assert len(synthetic_dataset) == 12

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
