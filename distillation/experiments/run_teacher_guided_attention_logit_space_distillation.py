"""Rerun attention-guidance experiments after the logit-space loss fix."""

from distillation.experiments.common import (
  default_teacher_manifest_path,
  format_condition_value,
  run_condition, log_results, load_distill_dataset, load_manifest,
  RunTimeConfig,
  EPOCHS, PATIENCE, LR, WD, BATCH_SIZE, DEVICE,
)
from distillation.losses import RelationalTGALoss, TeacherGuidedAttnLoss

TEACHER_MANIFEST = default_teacher_manifest_path('run_concat_HE_CD20_CD3_mlp__bs32')

CONDITIONS = [
  (
    "teacher_guided_attention_logit_space_"
    f"alpha{format_condition_value(0)}_"
    f"beta{format_condition_value(0)}_"
    f"gamma{format_condition_value(1)}_"
    f"delta{format_condition_value(0)}_"
    f"tau{format_condition_value(1.0)}",
    TeacherGuidedAttnLoss,
    dict(alpha=0, beta=0, gamma=1, delta=0, tau=1.0),
  ),
  (
    "teacher_guided_attention_logit_space_"
    f"alpha{format_condition_value(0)}_"
    f"beta{format_condition_value(0)}_"
    f"gamma{format_condition_value(1)}_"
    f"delta{format_condition_value(0)}_"
    f"tau{format_condition_value(0.5)}",
    TeacherGuidedAttnLoss,
    dict(alpha=0, beta=0, gamma=1, delta=0, tau=0.5),
  ),
  (
    "relational_teacher_guided_attention_logit_space_"
    f"gamma{format_condition_value(1)}_"
    f"lambda{format_condition_value(0)}_"
    f"tau{format_condition_value(1.0)}",
    RelationalTGALoss,
    dict(gamma=1, lam=0, tau=1.0),
  ),
  (
    "relational_teacher_guided_attention_logit_space_"
    f"gamma{format_condition_value(1)}_"
    f"lambda{format_condition_value(1)}_"
    f"tau{format_condition_value(1.0)}",
    RelationalTGALoss,
    dict(gamma=1, lam=1, tau=1.0),
  ),
  (
    "relational_teacher_guided_attention_logit_space_"
    f"gamma{format_condition_value(1)}_"
    f"lambda{format_condition_value(0.1)}_"
    f"tau{format_condition_value(1.0)}",
    RelationalTGALoss,
    dict(gamma=1, lam=0.1, tau=1.0),
  ),
]


def main():
  manifest = load_manifest(TEACHER_MANIFEST)
  dataset, intersection_names = load_distill_dataset(manifest)

  for cond_name, loss_cls, loss_kwargs in CONDITIONS:
    print(f"\n{'#'*70}\n# {cond_name}  {loss_kwargs}\n{'#'*70}")
    config = RunTimeConfig()
    config.training.epochs        = EPOCHS
    config.training.learning_rate = LR
    config.training.weight_decay  = WD
    config.training.patience      = PATIENCE
    config.training.batch_size    = BATCH_SIZE
    config.training.device        = DEVICE
    distill_loss = loss_cls(**loss_kwargs)
    results = run_condition(cond_name, config, distill_loss, manifest, dataset)
    log_results(
      {cond_name: results}, config=config, distill_loss=distill_loss,
      manifest=manifest, stains=intersection_names,
    )


if __name__ == '__main__':
  main()
