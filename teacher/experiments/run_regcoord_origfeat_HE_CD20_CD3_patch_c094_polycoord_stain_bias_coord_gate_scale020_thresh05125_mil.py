"""Run c094 polycoord stain-bias coordinate-gate scale 0.20 MIL."""

import os

from teacher.experiments.common import (
  run_condition,
  log_results,
  find_common_sample_keys,
)
from teacher.experiments.run_regcoord_origfeat_HE_CD20_CD3_patch_polycoord_stain_bias_coord_gate_attdim164_coorddim24_thresh05125_mil import (
  REGCOORD_PATCH_FEAT_ROOT,
  STAINS,
  MIN_ALIGNED_PATCHES,
  ALIGNMENT_MODE,
  CACHE_ALIGNED,
  N_RUNS,
  K_FOLDS,
  OUTPUTS_DIR,
  SHARED_LOG_FILE,
  make_config as make_c091_config,
)


CONDITION_NAME = os.path.splitext(os.path.basename(__file__))[0]
COORD_GATE_SCALE = 0.2


def make_config(common_keys):
  config = make_c091_config(common_keys)
  config.model.model_kwargs['coord_gate_scale'] = COORD_GATE_SCALE
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
