"""K-fold entry point for relational teacher-guided attention distillation."""

from distillation.experiments.common import (
  default_teacher_manifest_path,
  format_condition_value,
  run_condition, log_results, load_distill_dataset, load_manifest,
  RunTimeConfig,
  EPOCHS, PATIENCE, LR, WD, BATCH_SIZE, DEVICE,
)
from distillation.losses import RelationalTGALoss


# =============================================================================
# 配置区 — 修改此处
# =============================================================================

TEACHER_MANIFEST = default_teacher_manifest_path('run_concat_HE_CD20_CD3_mlp__bs32')

CONDITIONS = [
  (
    "relational_teacher_guided_attention_"
    f"gamma{format_condition_value(1)}_"
    f"lambda{format_condition_value(0)}_"
    f"tau{format_condition_value(1.0)}",
    1, 0, 1.0,
  ),
  (
    "relational_teacher_guided_attention_"
    f"gamma{format_condition_value(0)}_"
    f"lambda{format_condition_value(1)}_"
    f"tau{format_condition_value(1.0)}",
    0, 1, 1.0,
  ),
  (
    "relational_teacher_guided_attention_"
    f"gamma{format_condition_value(1)}_"
    f"lambda{format_condition_value(0.1)}_"
    f"tau{format_condition_value(1.0)}",
    1, 0.1, 1.0,
  ),
]


# =============================================================================
# 配置构建
# =============================================================================

def make_config(gamma, lam, tau) -> tuple[RunTimeConfig, RelationalTGALoss]:
  config = RunTimeConfig()
  config.training.epochs        = EPOCHS
  config.training.learning_rate = LR
  config.training.weight_decay  = WD
  config.training.patience      = PATIENCE
  config.training.batch_size    = BATCH_SIZE
  config.training.device        = DEVICE
  distill_loss = RelationalTGALoss(gamma=gamma, lam=lam, tau=tau)
  return config, distill_loss


# =============================================================================
# 主流程
# =============================================================================

def main():
  manifest = load_manifest(TEACHER_MANIFEST)
  dataset, intersection_names = load_distill_dataset(manifest)

  all_results = {}
  for cond_name, gamma, lam, tau in CONDITIONS:
    print(f"\n{'#'*70}\n# Condition: {cond_name}  (gamma={gamma}, lam={lam}, tau={tau})\n{'#'*70}")
    config, distill_loss = make_config(gamma, lam, tau)
    results = run_condition(cond_name, config, distill_loss, manifest, dataset)
    all_results[cond_name] = results
    log_results(
      {cond_name: results}, config=config, distill_loss=distill_loss,
      manifest=manifest, stains=intersection_names,
    )


if __name__ == '__main__':
  main()
