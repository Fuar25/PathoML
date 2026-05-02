"""Launch teacher experiment runs across GPUs and aggregate run metrics."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from teacher.experiments.common import (
  BASE_SEED,
  K_FOLDS,
  N_RUNS,
  OUTPUTS_DIR,
  SHARED_LOG_FILE,
  find_common_sample_keys,
  log_results,
  save_manifest,
)


DEFAULT_MODULES = [
  'run_regcoord_origfeat_HE_CD20_CD3_patch_concat_abmil',
  'run_regcoord_origfeat_HE_CD20_CD3_patch_fusion_mil',
]


def _module_name(name: str) -> str:
  return name if name.startswith('teacher.experiments.') else f'teacher.experiments.{name}'


def _condition_name(module: Any) -> str:
  return getattr(module, 'CONDITION_NAME', module.__name__.rsplit('.', 1)[-1])


def _condition_root(module: Any) -> str:
  for attr in (
    'REGCOORD_PATCH_FEAT_ROOT',
    'REGISTERED_PATCH_FEAT_ROOT',
    'REGISTERED_SLIDE_FEAT_ROOT',
    'SLIDE_FEAT_ROOT',
    'PATCH_FEAT_ROOT',
  ):
    if hasattr(module, attr):
      return getattr(module, attr)
  raise AttributeError(f"Cannot infer data root for {module.__name__}")


def _build_config(module: Any):
  stains = list(getattr(module, 'STAINS'))
  common_keys = find_common_sample_keys(_condition_root(module), stains)
  return module.make_config(common_keys)


def _run_metrics_path(condition_name: str, run_idx: int) -> Path:
  return Path(OUTPUTS_DIR) / condition_name / f'run_{run_idx:02d}' / 'run_metrics.json'


def _load_metrics(condition_name: str, n_runs: int) -> dict[str, list[float]]:
  run_means = []
  all_fold_aucs = []
  run_f1_means = []
  all_fold_f1s = []
  missing = []

  for run_idx in range(n_runs):
    path = _run_metrics_path(condition_name, run_idx)
    if not path.exists():
      missing.append(str(path))
      continue
    with path.open('r', encoding='utf-8') as f:
      payload = json.load(f)
    fold_aucs = [float(v) for v in payload['fold_aucs']]
    fold_f1s = [float(v) for v in payload['fold_f1s']]
    run_means.append(float(np.mean(fold_aucs)))
    all_fold_aucs.extend(fold_aucs)
    run_f1_means.append(float(np.mean(fold_f1s)))
    all_fold_f1s.extend(fold_f1s)

  if missing:
    raise RuntimeError("Missing run metric files:\n" + "\n".join(missing))
  return {
    'run_means': run_means,
    'all_fold_aucs': all_fold_aucs,
    'run_f1_means': run_f1_means,
    'all_fold_f1s': all_fold_f1s,
  }


def _launch_module(
  module_name: str,
  run_idx: int,
  gpu: str,
  n_runs: int,
  k_folds: int,
  base_seed: int,
) -> subprocess.Popen:
  env = os.environ.copy()
  env['CUDA_VISIBLE_DEVICES'] = gpu
  env['PATHOML_RUN_INDICES'] = str(run_idx)
  env['PATHOML_N_RUNS'] = str(n_runs)
  env['PATHOML_K_FOLDS'] = str(k_folds)
  env['PATHOML_BASE_SEED'] = str(base_seed)
  env['PATHOML_SKIP_CONDITION_LOG'] = '1'
  env['PATHOML_SKIP_MANIFEST'] = '1'
  env.setdefault('PYTHONUNBUFFERED', '1')
  command = [sys.executable, '-m', module_name]
  return subprocess.Popen(command, env=env)


def _run_parallel(
  module_name: str,
  gpus: list[str],
  n_runs: int,
  k_folds: int,
  base_seed: int,
) -> None:
  pending = list(range(n_runs))
  running: list[tuple[int, str, subprocess.Popen]] = []
  while pending or running:
    while pending and len(running) < len(gpus):
      run_idx = pending.pop(0)
      used_gpus = {gpu for _, gpu, _ in running}
      gpu = next(gpu for gpu in gpus if gpu not in used_gpus)
      print(f"launch {module_name} run_{run_idx:02d} on CUDA_VISIBLE_DEVICES={gpu}")
      running.append((
        run_idx,
        gpu,
        _launch_module(module_name, run_idx, gpu, n_runs, k_folds, base_seed),
      ))

    still_running = []
    for run_idx, gpu, proc in running:
      status = proc.poll()
      if status is None:
        still_running.append((run_idx, gpu, proc))
      elif status != 0:
        raise RuntimeError(f"{module_name} run_{run_idx:02d} failed with status {status}")
      else:
        print(f"done {module_name} run_{run_idx:02d} on gpu {gpu}")
    running = still_running
    if running:
      time.sleep(5)


def _aggregate(module_name: str, n_runs: int, k_folds: int, base_seed: int) -> None:
  module = importlib.import_module(module_name)
  condition_name = _condition_name(module)
  config = _build_config(module)
  results = _load_metrics(condition_name, n_runs)
  save_manifest(condition_name, config, n_runs, k_folds, OUTPUTS_DIR, base_seed)
  log_results(
    {condition_name: results},
    SHARED_LOG_FILE,
    config=config,
    n_runs=n_runs,
    k_folds=k_folds,
    base_seed=base_seed,
    stains=list(getattr(module, 'STAINS', [])),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--modules', nargs='+', default=DEFAULT_MODULES)
  parser.add_argument('--gpus', required=True, help='Comma-separated physical GPU IDs, e.g. 3,5')
  parser.add_argument('--n-runs', type=int, default=N_RUNS)
  parser.add_argument('--k-folds', type=int, default=K_FOLDS)
  parser.add_argument('--base-seed', type=int, default=BASE_SEED)
  parser.add_argument('--aggregate-only', action='store_true')
  args = parser.parse_args()

  gpus = [part.strip() for part in args.gpus.split(',') if part.strip()]
  if not gpus and not args.aggregate_only:
    raise ValueError('--gpus must include at least one GPU ID')

  for raw_module in args.modules:
    module_name = _module_name(raw_module)
    if not args.aggregate_only:
      _run_parallel(module_name, gpus, args.n_runs, args.k_folds, args.base_seed)
    _aggregate(module_name, args.n_runs, args.k_folds, args.base_seed)


if __name__ == '__main__':
  main()
