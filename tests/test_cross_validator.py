"""Integration tests: CrossValidator end-to-end with synthetic in-memory data."""

import math
import os

from PathoML.optimization.trainer import CrossValidator, Trainer


def _run_cv(synthetic_dataset, model_builder_fn, trainer_config, k_folds=2):
  strategy = CrossValidator(model_builder_fn, synthetic_dataset, trainer_config, k_folds=k_folds)
  return Trainer(strategy).fit()


def test_returns_correct_n_folds(synthetic_dataset, model_builder_fn, trainer_config):
  result = _run_cv(synthetic_dataset, model_builder_fn, trainer_config, k_folds=2)
  assert len(result.fold_results) == 2


def test_fold_result_numeric_fields(synthetic_dataset, model_builder_fn, trainer_config):
  result = _run_cv(synthetic_dataset, model_builder_fn, trainer_config)
  for fold in result.fold_results:
    # (1) Core metrics must be floats and not NaN
    assert isinstance(fold.test_loss, float) and not math.isnan(fold.test_loss)
    assert isinstance(fold.test_acc,  float) and not math.isnan(fold.test_acc)
    assert isinstance(fold.test_auc,  float) and not math.isnan(fold.test_auc)


def test_patient_metrics_computed(synthetic_dataset, model_builder_fn, trainer_config):
  result = _run_cv(synthetic_dataset, model_builder_fn, trainer_config)
  for fold in result.fold_results:
    # (1) Patient-level metrics must be populated (not None)
    assert fold.patient_acc is not None
    assert fold.patient_auc is not None
    assert fold.patient_f1 is not None


def test_checkpoints_saved(synthetic_dataset, model_builder_fn, trainer_config, tmp_path):
  # trainer_config already points save_dir at tmp_path via fixture
  _run_cv(synthetic_dataset, model_builder_fn, trainer_config)
  checkpoint_files = [
    f for f in os.listdir(str(tmp_path))
    if f.endswith('.pt') or f.endswith('.pth')
  ]
  assert len(checkpoint_files) > 0, "No checkpoint files found after CV"
