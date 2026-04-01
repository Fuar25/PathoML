"""Standard KD 蒸馏实验 K折 CV 入口。

蒸馏损失: L_total = L_task + alpha * L_feat + beta * L_kd
消融实验：修改下方 ALPHA/BETA/TEMPERATURE：
  - Baseline:   alpha=0, beta=0
  - +L_feat:    alpha=1, beta=0
  - +L_kd:      alpha=0, beta=1, temperature=4
  - Full:       alpha=1, beta=1, temperature=4
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
  run_condition, log_results, load_distill_dataset, load_manifest,
  RunTimeConfig,
  EPOCHS, PATIENCE, LR, WD, BATCH_SIZE, DEVICE,
)
from losses import StandardKDLoss


# =============================================================================
# 配置区 — 修改此处
# =============================================================================

TEACHER_MANIFEST = '/home/william/PycharmProjects/PathoML/runs/outputs/run_concat_HE_CD20_CD3_mlp/manifest.json'

# 蒸馏超参
ALPHA       = 0
BETA        = 1
TEMPERATURE = 4.0

_SCRIPT_NAME   = os.path.splitext(os.path.basename(__file__))[0]
CONDITION_NAME = f"{_SCRIPT_NAME}_a{ALPHA}b{BETA}T{TEMPERATURE}"


# =============================================================================
# 配置构建
# =============================================================================

def make_config() -> tuple[RunTimeConfig, StandardKDLoss]:
  config = RunTimeConfig()
  config.training.epochs        = EPOCHS
  config.training.learning_rate = LR
  config.training.weight_decay  = WD
  config.training.patience      = PATIENCE
  config.training.batch_size    = BATCH_SIZE
  config.training.device        = DEVICE
  distill_loss = StandardKDLoss(alpha=ALPHA, beta=BETA, temperature=TEMPERATURE)
  return config, distill_loss


# =============================================================================
# 主流程
# =============================================================================

def main():
  manifest = load_manifest(TEACHER_MANIFEST)
  dataset, intersection_names = load_distill_dataset(manifest)
  config, distill_loss = make_config()
  results = run_condition(CONDITION_NAME, config, distill_loss, manifest, dataset)
  log_results(
    {CONDITION_NAME: results}, config=config, distill_loss=distill_loss,
    manifest=manifest, sample_intersection=intersection_names,
  )


if __name__ == '__main__':
  main()
