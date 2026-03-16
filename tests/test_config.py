"""Tests for config dataclasses: instantiation and defaults."""

from PathoML.config.config import RunTimeConfig, ModelConfig, TrainingConfig, DatasetConfig, LoggingConfig


def test_default_instantiation():
  config = RunTimeConfig()
  assert config is not None
  assert config.dataset is not None
  assert config.model is not None
  assert config.training is not None
  assert config.logging is not None


def test_field_assignment():
  config = RunTimeConfig()
  config.training.epochs = 50
  assert config.training.epochs == 50


def test_nested_defaults():
  m = ModelConfig()
  assert m.input_dim == 1536
  assert m.num_classes == 1
  assert m.hidden_dim == 512
  assert m.dropout == 0.2


def test_model_kwargs_empty_by_default():
  assert ModelConfig().model_kwargs == {}


def test_training_config_defaults():
  t = TrainingConfig()
  assert t.batch_size == 1
  assert t.seed == 42
  assert 0.0 < t.learning_rate < 1.0


def test_runtime_config_independence():
  # (1) Two instances should be independent — no shared singleton state
  c1 = RunTimeConfig()
  c2 = RunTimeConfig()
  c1.training.epochs = 99
  assert c2.training.epochs != 99
