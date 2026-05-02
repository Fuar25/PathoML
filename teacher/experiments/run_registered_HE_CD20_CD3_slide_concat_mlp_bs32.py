"""Run MLP on registered HE/CD20/CD3 slide-concat features."""

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
  DROPOUT_RATE,
  SLIDE_LR,
  OUTPUTS_DIR,
  SHARED_LOG_FILE,
)


REGISTERED_SLIDE_FEAT_ROOT = os.environ.get(
  'PATHOML_REGISTERED_SLIDE_FEATURE_ROOT',
  '/home/sbh/Features/GigaPath-Slide-Feature-Reg',
)
MLP_HIDDEN_DIM = 128
STAINS = ['HE', 'CD20', 'CD3']
CONDITION_NAME = os.path.splitext(os.path.basename(__file__))[0]


def matched_sample_keys() -> set[tuple[str, str]]:
  registered_keys = find_common_sample_keys(REGISTERED_SLIDE_FEAT_ROOT, STAINS)
  original_keys = find_common_sample_keys(SLIDE_FEAT_ROOT, STAINS)
  return registered_keys & original_keys


def make_config(common_keys) -> RunTimeConfig:
  config = RunTimeConfig()
  config.dataset.dataset_name = 'MultimodalConcatSlideDataset'
  config.dataset.dataset_kwargs = {
    'data_root': REGISTERED_SLIDE_FEAT_ROOT,
    'modality_names': STAINS,
    'allowed_sample_keys': common_keys,
    'labels_csv': LABELS_CSV,
  }
  config.model.model_name = 'mlp'
  config.model.model_kwargs = {
    'hidden_dim': MLP_HIDDEN_DIM,
    'dropout': DROPOUT_RATE,
  }
  config.training.device = DEVICE
  config.training.epochs = EPOCHS
  config.training.patience = PATIENCE
  config.training.learning_rate = SLIDE_LR
  config.training.weight_decay = WD
  config.training.batch_size = 32
  return config


def main():
  common_keys = matched_sample_keys()
  print(f"matched registered/original slide samples ({' ∩ '.join(STAINS)}): {len(common_keys)}")
  print(f"registered slide root: {REGISTERED_SLIDE_FEAT_ROOT}")

  config = make_config(common_keys)
  results = run_condition(CONDITION_NAME, config, N_RUNS, K_FOLDS, output_dir=OUTPUTS_DIR)
  log_results(
    {CONDITION_NAME: results},
    SHARED_LOG_FILE,
    config=config,
    stains=STAINS,
  )


if __name__ == '__main__':
  main()
