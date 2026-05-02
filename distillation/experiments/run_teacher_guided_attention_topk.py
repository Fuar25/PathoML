"""K-fold entry point for top-k cosine teacher-guided attention distillation.

This keeps the cosine-derived teacher target from the active baseline but
restricts attention-logit supervision to the most salient teacher-selected
instances within each bag. The top-k ratio is configurable via --topk-ratio
so a single script covers the full ratio sweep without looping internally.

Usage:
  python run_teacher_guided_attention_topk.py --topk-ratio 0.25
"""

import argparse

from distillation.experiments.common import (
  default_teacher_manifest_path,
  build_condition_name,
  build_runtime_config,
  run_condition, log_results, load_distill_dataset, load_manifest,
)
from distillation.losses import (
  CompositeDistillationLoss,
  TaskLoss,
  TopKCosineAttentionLogitLoss,
)

TEACHER_MANIFEST = default_teacher_manifest_path('run_concat_HE_CD20_CD3_mlp_bs32')


def make_distill_loss(topk_ratio: float) -> CompositeDistillationLoss:
  return CompositeDistillationLoss([
    TaskLoss(),
    TopKCosineAttentionLogitLoss(topk_ratio=topk_ratio),
  ])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--topk-ratio', type=float, default=0.25,
                      help='Fraction of top teacher-selected patches to supervise (default: 0.25)')
  args = parser.parse_args()

  manifest = load_manifest(TEACHER_MANIFEST)
  dataset, intersection_names = load_distill_dataset(manifest)
  distill_loss = make_distill_loss(args.topk_ratio)
  cond_name = build_condition_name('teacher_guided_attention', distill_loss)
  print(f"\n{'#'*70}\n# {cond_name}  {distill_loss.describe()}\n{'#'*70}")
  config = build_runtime_config()
  results = run_condition(cond_name, config, distill_loss, manifest, dataset)
  log_results(
    {cond_name: results}, config=config, distill_loss=distill_loss,
    manifest=manifest, stains=intersection_names,
  )


if __name__ == '__main__':
  main()
