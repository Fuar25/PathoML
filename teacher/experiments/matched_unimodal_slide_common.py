"""Helpers for matched unimodal slide-level diagnostics."""

import os

from teacher.experiments.common import (
  run_condition,
  log_results,
  find_common_sample_keys,
  RunTimeConfig,
  SLIDE_FEAT_ROOT,
  LABELS_CSV,
  N_RUNS,
  K_FOLDS,
  DEVICE,
  EPOCHS,
  PATIENCE,
  WD,
  BATCH_SIZE,
  SLIDE_LR,
  OUTPUTS_DIR,
  SHARED_LOG_FILE,
)


REGISTERED_SLIDE_FEAT_ROOT = os.environ.get(
  'PATHOML_REGISTERED_SLIDE_FEATURE_ROOT',
  '/home/sbh/Features/GigaPath-Slide-Feature-Reg',
)
MATCHING_STAINS = ['HE', 'CD20', 'CD3']


def matched_sample_keys() -> set[tuple[str, str]]:
  registered_keys = find_common_sample_keys(REGISTERED_SLIDE_FEAT_ROOT, MATCHING_STAINS)
  original_keys = find_common_sample_keys(SLIDE_FEAT_ROOT, MATCHING_STAINS)
  return registered_keys & original_keys


def make_config(
  *,
  stain: str,
  data_root: str,
  common_keys: set[tuple[str, str]],
) -> RunTimeConfig:
  config = RunTimeConfig()
  config.dataset.dataset_name = 'UnimodalSlideDataset'
  config.dataset.dataset_kwargs['data_root'] = data_root
  config.dataset.dataset_kwargs['stain'] = stain
  config.dataset.dataset_kwargs['allowed_sample_keys'] = common_keys
  config.dataset.dataset_kwargs['labels_csv'] = LABELS_CSV
  config.model.model_name = 'linear_probe'
  config.model.model_kwargs = {}
  config.training.device = DEVICE
  config.training.epochs = EPOCHS
  config.training.patience = PATIENCE
  config.training.learning_rate = SLIDE_LR
  config.training.weight_decay = WD
  config.training.batch_size = BATCH_SIZE
  return config


def run_unimodal_slide_condition(
  *,
  condition_name: str,
  stain: str,
  data_root: str,
) -> None:
  common_keys = matched_sample_keys()
  print(f"matched registered/original slide samples ({' ∩ '.join(MATCHING_STAINS)}): {len(common_keys)}")
  print(f"stain: {stain}")
  print(f"data root: {data_root}")

  config = make_config(
    stain=stain,
    data_root=data_root,
    common_keys=common_keys,
  )
  results = run_condition(condition_name, config, N_RUNS, K_FOLDS, output_dir=OUTPUTS_DIR)
  log_results(
    {condition_name: results},
    SHARED_LOG_FILE,
    config=config,
    stains=MATCHING_STAINS,
  )
