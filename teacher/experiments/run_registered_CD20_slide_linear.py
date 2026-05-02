"""Run linear probe on registered CD20 slide features."""

import os

from teacher.experiments.matched_unimodal_slide_common import (
  REGISTERED_SLIDE_FEAT_ROOT,
  run_unimodal_slide_condition,
)


STAIN = 'CD20'
CONDITION_NAME = os.path.splitext(os.path.basename(__file__))[0]


def main():
  run_unimodal_slide_condition(
    condition_name=CONDITION_NAME,
    stain=STAIN,
    data_root=REGISTERED_SLIDE_FEAT_ROOT,
  )


if __name__ == '__main__':
  main()
