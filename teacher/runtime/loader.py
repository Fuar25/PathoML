"""Teacher-specific runtime module loader."""

from __future__ import annotations

import importlib


_TEACHER_DATASET_MODULES = [
  'teacher.dataset.unimodal_patch',
  'teacher.dataset.unimodal_slide',
  'teacher.dataset.multimodal_concat_slide',
  'teacher.dataset.multimodal_fusion_slide',
  'teacher.dataset.registered_multimodal_patch',
]

_TEACHER_MODEL_MODULES = [
  'teacher.models.abmil',
  'teacher.models.linear_probe',
  'teacher.models.mlp',
  'teacher.models.registered_patch_fusion',
]


def load_teacher_modules() -> None:
  """Import teacher concrete datasets/models so they register themselves."""
  for module_path in _TEACHER_DATASET_MODULES + _TEACHER_MODEL_MODULES:
    importlib.import_module(module_path)
