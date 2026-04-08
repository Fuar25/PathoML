"""K-fold entry point for teacher-guided attention distillation."""

from distillation.experiments.common import (
  default_teacher_manifest_path,
  format_condition_value,
  run_condition, log_results, load_distill_dataset, load_manifest,
  RunTimeConfig,
  EPOCHS, PATIENCE, LR, WD, BATCH_SIZE,
)
from distillation.losses import TeacherGuidedAttnLoss


# =============================================================================
# 配置区 — 修改此处
# =============================================================================

TEACHER_MANIFEST = default_teacher_manifest_path('run_concat_HE_CD20_CD3_mlp__bs32')

ALPHA       = 0
BETA        = 0
GAMMA       = 1
DELTA       = 0
TEMPERATURE = 4.0
TAU         = 0.5

CONDITION_NAME = (
  "teacher_guided_attention_"
  f"alpha{format_condition_value(ALPHA)}_"
  f"beta{format_condition_value(BETA)}_"
  f"gamma{format_condition_value(GAMMA)}_"
  f"delta{format_condition_value(DELTA)}_"
  f"temperature{format_condition_value(TEMPERATURE)}_"
  f"tau{format_condition_value(TAU)}"
)


# =============================================================================
# 配置构建
# =============================================================================

def make_config() -> tuple[RunTimeConfig, TeacherGuidedAttnLoss]:
  config = RunTimeConfig()
  config.training.epochs        = EPOCHS
  config.training.learning_rate = LR
  config.training.weight_decay  = WD
  config.training.patience      = PATIENCE
  config.training.batch_size    = BATCH_SIZE
  config.training.device        = 'cuda:1'
  distill_loss = TeacherGuidedAttnLoss(
    alpha=ALPHA, beta=BETA, temperature=TEMPERATURE,
    gamma=GAMMA, delta=DELTA, tau=TAU,
  )
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
