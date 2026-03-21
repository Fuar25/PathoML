"""Tests for EarlyStopping in optimization.trainer."""

import os
import warnings

import pytest
import torch
import torch.nn as nn

from PathoML.optimization.trainer import EarlyStopping


# ---------------------------------------------------------------------------
# EarlyStopping — patience logic
# ---------------------------------------------------------------------------

def test_early_stopping_improves(tmp_path):
  model = nn.Linear(4, 2)
  es = EarlyStopping(patience=3, model=model, ckpt_path=str(tmp_path / "ckpt.pt"))
  assert es.step(0.5, 1) == False  # first: always improvement (> -inf)
  assert es.step(0.7, 2) == False  # still improving (higher AUC)


def test_early_stopping_patience_exhausted(tmp_path):
  model = nn.Linear(4, 2)
  es = EarlyStopping(patience=3, model=model, ckpt_path=str(tmp_path / "ckpt.pt"))
  es.step(0.9, 1)   # best = 0.9
  es.step(0.8, 2)   # counter = 1 (worse)
  es.step(0.7, 3)   # counter = 2
  assert es.step(0.6, 4) == True  # counter = 3 >= patience → stop


def test_early_stopping_does_not_stop_early(tmp_path):
  model = nn.Linear(4, 2)
  es = EarlyStopping(patience=3, model=model, ckpt_path=str(tmp_path / "ckpt.pt"))
  es.step(0.9, 1)
  es.step(0.8, 2)   # counter = 1
  # counter = 2, not yet at patience=3
  assert es.step(0.7, 3) == False


def test_early_stopping_best_epoch_tracked(tmp_path):
  model = nn.Linear(4, 2)
  es = EarlyStopping(patience=5, model=model, ckpt_path=str(tmp_path / "ckpt.pt"))
  es.step(0.5, 1)
  es.step(0.8, 2)   # new best at epoch 2 (higher AUC)
  es.step(0.7, 3)   # no improvement
  assert es.best_epoch == 2


def test_early_stopping_reset(tmp_path):
  model = nn.Linear(4, 2)
  es = EarlyStopping(patience=3, model=model, ckpt_path=str(tmp_path / "ckpt.pt"))
  es.step(0.8, 1)
  es.step(0.7, 2)
  es.reset()
  assert es.best_val_auc == float('-inf')
  assert es.patience_counter == 0
  assert es.best_epoch == 0


# ---------------------------------------------------------------------------
# EarlyStopping — checkpoint save / load
# ---------------------------------------------------------------------------

def test_early_stopping_saves_checkpoint_on_improvement(tmp_path):
  model = nn.Linear(4, 2)
  ckpt_path = str(tmp_path / "best.pt")
  es = EarlyStopping(patience=3, model=model, ckpt_path=ckpt_path)
  es.step(0.8, 1)   # improvement (> -inf), should save
  assert os.path.exists(ckpt_path)


def test_early_stopping_does_not_save_on_no_improvement(tmp_path):
  model = nn.Linear(4, 2)
  ckpt_path = str(tmp_path / "best.pt")
  es = EarlyStopping(patience=3, model=model, ckpt_path=ckpt_path)
  es.step(0.8, 1)   # saves
  mtime_after_first = os.path.getmtime(ckpt_path)
  import time; time.sleep(0.05)
  es.step(0.5, 2)   # no improvement (lower AUC) — should NOT overwrite
  assert os.path.getmtime(ckpt_path) == mtime_after_first


def test_early_stopping_load_best_restores_weights(tmp_path):
  model = nn.Linear(4, 2)
  original_weight = model.weight.data.clone()
  ckpt_path = str(tmp_path / "best.pt")

  es = EarlyStopping(patience=3, model=model, ckpt_path=ckpt_path)
  es.step(0.8, 1)   # saves original weights

  # Perturb weights, then restore
  model.weight.data.fill_(0.0)
  assert not torch.allclose(model.weight.data, original_weight)

  es.load_best()
  assert torch.allclose(model.weight.data, original_weight)


def test_early_stopping_load_weights_only(tmp_path):
  # (1) weights_only=True should suppress FutureWarning about pickle
  model = nn.Linear(4, 2)
  ckpt_path = str(tmp_path / "best.pt")
  es = EarlyStopping(patience=3, model=model, ckpt_path=ckpt_path)
  es.step(0.8, 1)

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    es.load_best()

  future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
  assert len(future_warnings) == 0
