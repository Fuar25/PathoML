"""Integration tests: FullDatasetTrainer end-to-end with synthetic in-memory data."""

import os

from PathoML.optimization.trainer import FullDatasetTrainer, Trainer, TrainingResult


def _run_full(synthetic_dataset, model_builder_fn, trainer_config):
  strategy = FullDatasetTrainer(model_builder_fn, synthetic_dataset, trainer_config)
  return Trainer(strategy).fit()


def test_returns_training_result(synthetic_dataset, model_builder_fn, trainer_config):
  result = _run_full(synthetic_dataset, model_builder_fn, trainer_config)
  assert isinstance(result, TrainingResult)
  assert result.strategy_name == "FullDataset Training"
  assert result.fold_results == []   # full training has no fold splits


def test_checkpoint_saved(synthetic_dataset, model_builder_fn, trainer_config, tmp_path):
  # trainer_config already points save_dir at tmp_path via fixture
  _run_full(synthetic_dataset, model_builder_fn, trainer_config)
  assert os.path.exists(str(tmp_path / "model_deployment.pth"))
