"""Run linear probe on original CD3 slide features using registered samples."""

import os

from teacher.experiments.common import SLIDE_FEAT_ROOT
from teacher.experiments.matched_unimodal_slide_common import run_unimodal_slide_condition


STAIN = 'CD3'
CONDITION_NAME = os.path.splitext(os.path.basename(__file__))[0]


def main():
  run_unimodal_slide_condition(
    condition_name=CONDITION_NAME,
    stain=STAIN,
    data_root=SLIDE_FEAT_ROOT,
  )


if __name__ == '__main__':
  main()
