"""Run polycoord stain-bias coordinate-gate fusion MIL on RegCoord original HE/CD20/CD3 patch features."""

import os

from teacher.experiments.common import (
  run_condition,
  log_results,
  configure_fast_patch_training,
  env_bool,
  find_common_sample_keys,
  RunTimeConfig,
  LABELS_CSV,
  N_RUNS,
  K_FOLDS,
  DEVICE,
  EPOCHS,
  PATIENCE,
  LR,
  WD,
  DROPOUT_RATE,
  BATCH_SIZE,
  OUTPUTS_DIR,
  SHARED_LOG_FILE,
)


REGCOORD_PATCH_FEAT_ROOT = os.environ.get(
  'PATHOML_REGCOORD_PATCH_FEATURE_ROOT',
  '/home/sbh/Features/GigaPath-Patch-Feature-RegCoordOrigFeat',
)
STAINS = ['HE', 'CD20', 'CD3']
MIN_ALIGNED_PATCHES = int(os.environ.get('PATHOML_MIN_ALIGNED_PATCHES', '1'))
ALIGNMENT_MODE = os.environ.get('PATHOML_ALIGNMENT_MODE', 'union')
CACHE_ALIGNED = env_bool('PATHOML_CACHE_ALIGNED', True)
CONDITION_NAME = os.path.splitext(os.path.basename(__file__))[0]

MODALITY_HIDDEN_DIM = int(os.environ.get('PATHOML_MODALITY_HIDDEN_DIM', '256'))
FUSION_HIDDEN_DIM = int(os.environ.get('PATHOML_FUSION_HIDDEN_DIM', '256'))
FUSION_DROPOUT = float(os.environ.get('PATHOML_FUSION_DROPOUT', str(DROPOUT_RATE)))
MODALITY_DROPOUT = float(os.environ.get('PATHOML_MODALITY_DROPOUT', '0.1'))
COORD_HIDDEN_DIM = int(os.environ.get('PATHOML_COORD_HIDDEN_DIM', '24'))
ATTENTION_DIM = int(os.environ.get('PATHOML_ATTENTION_DIM', '164'))


def make_config(common_keys) -> RunTimeConfig:
  config = RunTimeConfig()
  config.dataset.dataset_name = 'RegisteredMultimodalPatchDataset'
  config.dataset.dataset_kwargs = {
    'data_root': REGCOORD_PATCH_FEAT_ROOT,
    'modality_names': STAINS,
    'allowed_sample_keys': common_keys,
    'labels_csv': LABELS_CSV,
    'min_aligned_patches': MIN_ALIGNED_PATCHES,
    'alignment_mode': ALIGNMENT_MODE,
    'cache_aligned': CACHE_ALIGNED,
  }
  config.model.model_name = 'registered_patch_polycoord_stain_bias_coord_gate_fusion_mil'
  config.model.model_kwargs = {
    'num_modalities': len(STAINS),
    'modality_hidden_dim': MODALITY_HIDDEN_DIM,
    'hidden_dim': FUSION_HIDDEN_DIM,
    'dropout': FUSION_DROPOUT,
    'modality_dropout': MODALITY_DROPOUT,
    'use_modality_mask_features': True,
    'coord_hidden_dim': COORD_HIDDEN_DIM,
    'attention_dim': ATTENTION_DIM,
  }
  config.training.device = DEVICE
  config.training.epochs = EPOCHS
  config.training.patience = PATIENCE
  config.training.learning_rate = LR
  config.training.weight_decay = WD
  config.training.batch_size = BATCH_SIZE
  config.training.patient_threshold = 0.5125
  configure_fast_patch_training(config)
  return config


def main():
  common_keys = find_common_sample_keys(REGCOORD_PATCH_FEAT_ROOT, STAINS)
  print(f"RegCoord original-feature common samples ({' + '.join(STAINS)}): {len(common_keys)}")
  print(f"min_aligned_patches: {MIN_ALIGNED_PATCHES}")
  print(f"alignment_mode: {ALIGNMENT_MODE}")
  print(f"cache_aligned: {CACHE_ALIGNED}")

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
