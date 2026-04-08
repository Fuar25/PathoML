"""K-fold entry point for multihead contrastive distillation."""

from distillation.experiments.common import (
  default_teacher_manifest_path,
  format_condition_value,
  run_condition, log_results, load_distill_dataset, load_manifest,
  RunTimeConfig,
  EPOCHS, PATIENCE, LR, WD, BATCH_SIZE, DEVICE, STUDENT_KWARGS,
)
from distillation.losses import RelationalTGALoss
from distillation.models.student import StudentTransABMIL_MH


# =============================================================================
# 配置区
# =============================================================================

TEACHER_MANIFEST = default_teacher_manifest_path('run_concat_HE_CD20_CD3_mlp__bs32')

MH_STUDENT_KWARGS = dict(
  **STUDENT_KWARGS,
  pool_heads=4,
)

CONDITIONS = [
  (
    "multihead_contrastive_distillation_"
    f"gamma{format_condition_value(0)}_"
    f"lambda{format_condition_value(1)}_"
    f"tau{format_condition_value(1.0)}_"
    f"pool_heads{format_condition_value(MH_STUDENT_KWARGS['pool_heads'])}",
    0, 1, 1.0,
  ),
  (
    "multihead_contrastive_distillation_"
    f"gamma{format_condition_value(0)}_"
    f"lambda{format_condition_value(0.1)}_"
    f"tau{format_condition_value(1.0)}_"
    f"pool_heads{format_condition_value(MH_STUDENT_KWARGS['pool_heads'])}",
    0, 0.1, 1.0,
  ),
]


# =============================================================================
# 主流程
# =============================================================================

def main():
  manifest = load_manifest(TEACHER_MANIFEST)
  dataset, intersection_names = load_distill_dataset(manifest)

  for cond_name, gamma, lam, tau in CONDITIONS:
    print(f"\n{'#'*70}\n# {cond_name}  (gamma={gamma}, lam={lam}, tau={tau})\n{'#'*70}")
    config = RunTimeConfig()
    config.training.epochs        = EPOCHS
    config.training.learning_rate = LR
    config.training.weight_decay  = WD
    config.training.patience      = PATIENCE
    config.training.batch_size    = BATCH_SIZE
    config.training.device        = DEVICE
    distill_loss = RelationalTGALoss(gamma=gamma, lam=lam, tau=tau)
    results = run_condition(
      cond_name, config, distill_loss, manifest, dataset,
      student_builder=lambda: StudentTransABMIL_MH(**MH_STUDENT_KWARGS),
    )
    log_results(
      {cond_name: results}, config=config, distill_loss=distill_loss,
      manifest=manifest, stains=intersection_names,
      student_kwargs=MH_STUDENT_KWARGS,
    )


if __name__ == '__main__':
  main()
