"""Shared pytest fixtures for PathoML unit tests."""

import sys
import os

# Ensure project root is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from PathoML.registry import Registry
from PathoML.interfaces import BaseDataset


@pytest.fixture
def fresh_registry():
  """Return a fresh Registry instance per test to avoid global state pollution."""
  return Registry("test")


@pytest.fixture
def synthetic_abmil_data():
  """Synthetic DataDict for ABMIL: features (1, 10, 64), label 0."""
  return {
    'features': torch.randn(1, 10, 64),
    'label': torch.tensor(0),
  }


# ---------------------------------------------------------------------------
# Integration test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dataset():
  """In-memory dataset satisfying the full BaseDataset contract.

  Layout: 40 samples, 20 patients (2 samples each), binary labels.
  10 negative patients (P0-P9) + 10 positive patients (P10-P19).
  Sufficient for stratified patient-level K-fold and train/val splits.
  """
  class SyntheticDataset(BaseDataset):
    def __init__(self):
      n_patients, samples_per = 20, 2
      self.n_instances = 5
      self.n_features = 32
      self._patient_ids = [
        f"P{i}" for i in range(n_patients) for _ in range(samples_per)
      ]
      self._labels = [
        0 if i < n_patients // 2 else 1
        for i in range(n_patients)
        for _ in range(samples_per)
      ]
      self.classes = {0, 1}
      self.data = [{'label': lbl} for lbl in self._labels]

    def __len__(self):
      return len(self._labels)

    def __getitem__(self, idx):
      return {
        'features': torch.randn(self.n_instances, self.n_features),
        'label': torch.tensor(self._labels[idx], dtype=torch.float32),
        'slide_id': f"sample_{idx}",
        'patient_id': self._patient_ids[idx],
      }

    def get_patient_ids(self):
      return self._patient_ids

    def get_labels(self):
      return list(self._labels)

  return SyntheticDataset()


@pytest.fixture
def model_builder_fn():
  """Factory that returns a fresh ABMIL instance (input_dim=32, hidden_dim=16)."""
  from PathoML.models.abmil import ABMIL
  return lambda: ABMIL(input_dim=32, hidden_dim=16, num_classes=1, dropout=0.0)


@pytest.fixture
def synthetic_varlen_dataset():
  """In-memory dataset with variable-length bags (N differs per sample).

  Same patient/label layout as synthetic_dataset, but N is random in [3, 10].
  Suitable for testing batch_size > 1 with ABMIL.
  """
  class SyntheticVarlenDataset(BaseDataset):
    def __init__(self):
      n_patients, samples_per = 20, 2
      self.n_features = 32
      self._patient_ids = [
        f"P{i}" for i in range(n_patients) for _ in range(samples_per)
      ]
      self._labels = [
        0 if i < n_patients // 2 else 1
        for i in range(n_patients)
        for _ in range(samples_per)
      ]
      self.classes = {0, 1}
      self._rng = torch.Generator().manual_seed(0)

    def __len__(self):
      return len(self._labels)

    def __getitem__(self, idx):
      n_instances = torch.randint(3, 11, (1,), generator=self._rng).item()
      return {
        'features': torch.randn(n_instances, self.n_features),
        'coords': torch.randn(n_instances, 2),
        'label': torch.tensor(self._labels[idx], dtype=torch.float32),
        'slide_id': f"sample_{idx}",
        'patient_id': self._patient_ids[idx],
      }

    def get_patient_ids(self):
      return self._patient_ids

    def get_labels(self):
      return list(self._labels)

  return SyntheticVarlenDataset()


@pytest.fixture
def trainer_config(tmp_path):
  """Minimal RunTimeConfig for integration tests: CPU, 1 epoch, tmp checkpoint dir."""
  from PathoML.config.config import RunTimeConfig
  config = RunTimeConfig()
  config.training.device = 'cpu'
  config.training.epochs = 1
  config.training.patience = 1
  config.training.batch_size = 1
  config.training.learning_rate = 1e-3
  config.training.seed = 42
  config.logging.save_dir = str(tmp_path)
  return config
