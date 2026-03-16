"""Optimization package: training strategies, registry, and base interfaces."""

from .interfaces import BaseDataset, BaseModel, BaseMIL, Aggregator, Classifier
from .patient_aggregation import aggregate_patient_predictions
from .registry import (
  register_model,
  register_dataset,
  create_model,
  create_dataset,
  load_runtime_plugins,
)

__all__ = [
  'BaseDataset',
  'BaseModel',
  'BaseMIL',
  'Aggregator',
  'Classifier',
  'aggregate_patient_predictions',
  'register_model',
  'register_dataset',
  'create_model',
  'create_dataset',
  'load_runtime_plugins',
]
