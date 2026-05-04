"""Tests for distillation experiment helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from distillation.experiments import common


def test_run_indices_from_env_defaults_to_all_runs(monkeypatch):
  monkeypatch.delenv('PATHOML_RUN_INDICES', raising=False)

  assert common._run_indices_from_env(3) == [0, 1, 2]


def test_run_indices_from_env_validates_bounds(monkeypatch):
  monkeypatch.setenv('PATHOML_RUN_INDICES', '0,2,5')

  with pytest.raises(ValueError, match='Invalid PATHOML_RUN_INDICES'):
    common._run_indices_from_env(3)


def test_log_results_can_skip_shared_log(monkeypatch, tmp_path):
  monkeypatch.setenv('PATHOML_SKIP_CONDITION_LOG', '1')
  log_path = tmp_path / 'results_log_mil_abmil.txt'

  common.log_results(
    {
      'candidate': {
        'run_means': [0.8],
        'all_fold_aucs': [0.7, 0.9],
        'all_fold_f1s': [0.6, 0.8],
      },
    },
    log_path=str(log_path),
  )

  assert not log_path.exists()


def test_run_condition_honors_run_indices_and_writes_metrics(monkeypatch, tmp_path):
  monkeypatch.setenv('PATHOML_RUN_INDICES', '0,2')

  def fake_run_distill_cv(
    dataset,
    config,
    distill_loss,
    teacher_ckpt_tmpl,
    k_folds,
    student_kwargs,
    student_builder,
  ):
    run_offset = config.training.seed - 100
    assert teacher_ckpt_tmpl == f'run_{run_offset:02d}/model_fold_{{fold}}_best.pth'
    return [0.70 + run_offset, 0.80 + run_offset], [0.60 + run_offset, 0.65 + run_offset]

  monkeypatch.setattr(common, 'run_distill_cv', fake_run_distill_cv)

  manifest = SimpleNamespace(
    n_runs=4,
    base_seed=100,
    k_folds=2,
    ckpt_tmpl='run_{run:02d}/model_fold_{fold}_best.pth',
  )
  config = common.build_runtime_config(device='cpu')

  results = common.run_condition(
    'candidate',
    config,
    distill_loss=object(),
    manifest=manifest,
    dataset=object(),
    output_dir=str(tmp_path),
  )

  assert results['run_indices'] == [0, 2]
  assert results['run_means'] == [0.75, 2.75]
  assert results['run_f1_means'] == [0.625, 2.625]

  run0_metrics = json.loads((tmp_path / 'candidate' / 'run_00' / 'run_metrics.json').read_text())
  run2_metrics = json.loads((tmp_path / 'candidate' / 'run_02' / 'run_metrics.json').read_text())

  assert run0_metrics == {
    'run_index': 0,
    'seed': 100,
    'fold_aucs': [0.7, 0.8],
    'fold_f1s': [0.6, 0.65],
    'run_auc_mean': 0.75,
    'run_f1_mean': 0.625,
  }
  assert run2_metrics == {
    'run_index': 2,
    'seed': 102,
    'fold_aucs': [2.7, 2.8],
    'fold_f1s': [2.6, 2.65],
    'run_auc_mean': 2.75,
    'run_f1_mean': 2.625,
  }
