"""Tests for EarlyStopping and CheckpointManager in optimization.trainer."""

import os
import warnings

import pytest
import torch
import torch.nn as nn

from PathoML.optimization.trainer import EarlyStopping, CheckpointManager


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

def test_early_stopping_improves():
  es = EarlyStopping(patience=3)
  assert es.step(1.0, 1) == False  # first: always improvement
  assert es.step(0.8, 2) == False  # still improving


def test_early_stopping_patience_exhausted():
  es = EarlyStopping(patience=3)
  es.step(1.0, 1)   # best = 1.0
  es.step(1.1, 2)   # counter = 1
  es.step(1.2, 3)   # counter = 2
  assert es.step(1.3, 4) == True  # counter = 3 >= patience → stop


def test_early_stopping_does_not_stop_early():
  es = EarlyStopping(patience=3)
  es.step(1.0, 1)
  es.step(1.1, 2)
  # counter = 2, not yet at patience=3
  assert es.step(1.2, 3) == False


def test_early_stopping_best_epoch_tracked():
  es = EarlyStopping(patience=5)
  es.step(1.0, 1)
  es.step(0.8, 2)   # new best at epoch 2
  es.step(0.9, 3)   # no improvement
  assert es.best_epoch == 2


def test_early_stopping_reset():
  es = EarlyStopping(patience=3)
  es.step(1.0, 1)
  es.step(1.1, 2)
  es.reset()
  assert es.best_val_loss == float('inf')
  assert es.patience_counter == 0
  assert es.best_epoch == 0


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

def test_checkpoint_save_creates_file(tmp_path):
  model = nn.Linear(4, 2)
  ckpt = CheckpointManager(str(tmp_path))
  ckpt.save(model, "test.pt")
  assert os.path.exists(str(tmp_path / "test.pt"))


def test_checkpoint_save_load_roundtrip(tmp_path):
  model = nn.Linear(4, 2)
  original_weight = model.weight.data.clone()

  ckpt = CheckpointManager(str(tmp_path))
  ckpt.save(model, "test.pt")

  # Perturb model weights, then reload
  model.weight.data.fill_(0.0)
  assert not torch.allclose(model.weight.data, original_weight)

  ckpt.load(model, "test.pt")
  assert torch.allclose(model.weight.data, original_weight)


def test_checkpoint_load_weights_only(tmp_path):
  # (1) weights_only=True should suppress FutureWarning about pickle
  model = nn.Linear(4, 2)
  ckpt = CheckpointManager(str(tmp_path))
  ckpt.save(model, "test.pt")

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ckpt.load(model, "test.pt")

  future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
  assert len(future_warnings) == 0


def test_checkpoint_manager_creates_dir(tmp_path):
  subdir = str(tmp_path / "deep" / "subdir")
  ckpt = CheckpointManager(subdir)
  assert os.path.isdir(subdir)
