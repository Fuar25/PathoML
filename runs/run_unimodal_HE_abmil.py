# HE 单模态 ABMIL 实验（GigaPath-Patch-Feature）。

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
  run_condition, log_results, find_common_sample_keys, modality_names,
  RunTimeConfig,
  HE_PATCH_BASE, CD20_SLIDE_BASE, CD3_SLIDE_BASE,
  N_RUNS, K_FOLDS, DEVICE, EPOCHS, PATIENCE, LR, WD, MLP_HIDDEN_DIM, DROPOUT_RATE,
  OUTPUTS_DIR, SHARED_LOG_FILE,
)

CONDITION_NAME = os.path.splitext(os.path.basename(__file__))[0]


def make_config(common_keys) -> RunTimeConfig:
  config = RunTimeConfig()
  config.dataset.dataset_name = "UnimodalSlideDataset"
  config.dataset.dataset_kwargs["data_path"] = HE_PATCH_BASE
  config.dataset.dataset_kwargs["allowed_sample_keys"] = common_keys
  config.model.model_name = "abmil"
  config.model.model_kwargs = {"hidden_dim": MLP_HIDDEN_DIM, "dropout": DROPOUT_RATE}
  config.training.device = DEVICE
  config.training.epochs = EPOCHS
  config.training.patience = PATIENCE
  config.training.learning_rate = LR
  config.training.weight_decay = WD
  return config


def main():
  # 与其他实验保持一致：取 HE patch 与 CD20, CD3 slide 的交集样本
  intersection_bases = [HE_PATCH_BASE, CD20_SLIDE_BASE, CD3_SLIDE_BASE]
  common_keys = find_common_sample_keys(intersection_bases)
  print(f"公共样本数（HE_patch ∩ CD20 ∩ CD3）: {len(common_keys)}")

  config = make_config(common_keys)
  results = run_condition(CONDITION_NAME, config, N_RUNS, K_FOLDS, output_dir=OUTPUTS_DIR)
  log_results({CONDITION_NAME: results}, SHARED_LOG_FILE, config=config,
              sample_intersection=modality_names(intersection_bases))


if __name__ == "__main__":
  main()
