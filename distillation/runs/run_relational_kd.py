"""Relational KD 蒸馏实验 K折 CV 入口。

蒸馏损失: L_total = L_task + gamma_d * L_dist + gamma_a * L_angle
RKD 匹配样本间的距离/角度结构而非单个表示，不依赖表示空间对齐。
需要 batch_size > 1。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
  run_condition, log_results, load_distill_dataset, load_manifest,
  RunTimeConfig,
  EPOCHS, PATIENCE, LR, WD, BATCH_SIZE, DEVICE,
)
from losses import RKDLoss


# =============================================================================
# 配置区 — 修改此处
# =============================================================================

TEACHER_MANIFEST = '/home/william/PycharmProjects/PathoML/runs/outputs/run_concat_HE_CD20_CD3_mlp/manifest.json'

# 蒸馏超参
GAMMA_D = 1      # L_dist 权重（distance-wise）
GAMMA_A = 2      # L_angle 权重（angle-wise）

_SCRIPT_NAME   = os.path.splitext(os.path.basename(__file__))[0]
CONDITION_NAME = f"{_SCRIPT_NAME}_d{GAMMA_D}a{GAMMA_A}"


# =============================================================================
# 配置构建
# =============================================================================

def make_config() -> tuple[RunTimeConfig, RKDLoss]:
  config = RunTimeConfig()
  config.training.epochs        = EPOCHS
  config.training.learning_rate = LR
  config.training.weight_decay  = WD
  config.training.patience      = PATIENCE
  config.training.batch_size    = BATCH_SIZE
  config.training.device        = DEVICE
  distill_loss = RKDLoss(gamma_d=GAMMA_D, gamma_a=GAMMA_A)
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
