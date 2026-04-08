"""K-fold entry point for relational knowledge distillation."""

from distillation.experiments.common import (
  default_teacher_manifest_path,
  format_condition_value,
  run_condition, log_results, load_distill_dataset, load_manifest,
  RunTimeConfig,
  EPOCHS, PATIENCE, LR, WD, BATCH_SIZE, DEVICE,
)
from distillation.losses import RKDLoss


# =============================================================================
# 配置区 — 修改此处
# =============================================================================

TEACHER_MANIFEST = default_teacher_manifest_path('run_concat_HE_CD20_CD3_mlp')

GAMMA_D = 1      # L_dist 权重（distance-wise）
GAMMA_A = 2      # L_angle 权重（angle-wise）

CONDITION_NAME = (
  "relational_knowledge_distillation_"
  f"gamma_distance{format_condition_value(GAMMA_D)}_"
  f"gamma_angle{format_condition_value(GAMMA_A)}"
)


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
    manifest=manifest, stains=intersection_names,
  )


if __name__ == '__main__':
  main()
