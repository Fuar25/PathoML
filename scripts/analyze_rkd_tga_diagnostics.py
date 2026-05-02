"""Diagnostic comparison for RKD vs RKD+TGA predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / 'distillation' / 'experiments' / 'outputs'
TEACHER_ROOT = ROOT / 'teacher' / 'experiments' / 'outputs' / 'run_concat_HE_CD20_CD3_mlp_bs32_lr4em4'
REPORT_PATH = ROOT / 'distillation' / 'experiments' / 'rkd_tga_diagnostic.md'


@dataclass(frozen=True)
class MethodSpec:
  name: str
  path: Path


METHODS = [
  MethodSpec('RKD', OUTPUT_ROOT / 'rkd_task_rkd_distance_rkd_angle_2p0'),
  MethodSpec(
    'RKD+TGA_0.5',
    OUTPUT_ROOT / 'rkd_tga_task_rkd_distance_rkd_angle_2p0_attn_cosine_logits_no_detach_0p5',
  ),
  MethodSpec(
    'RKD+TGA_0.75',
    OUTPUT_ROOT / 'rkd_tga_task_rkd_distance_rkd_angle_2p0_attn_cosine_logits_no_detach_0p75',
  ),
]


def _read_predictions(path: Path, run_idx: int) -> pd.DataFrame:
  csv_path = path / f'run_{run_idx:02d}' / 'cv_predictions.csv'
  df = pd.read_csv(csv_path)
  keep = ['patient_id', 'patient_label', 'patient_prob', 'patient_pred']
  df = df[keep].drop_duplicates('patient_id').copy()
  df['run'] = run_idx
  return df


def _read_method(spec: MethodSpec) -> pd.DataFrame:
  frames = []
  for run_idx in range(5):
    df = _read_predictions(spec.path, run_idx)
    df['method'] = spec.name
    frames.append(df)
  return pd.concat(frames, ignore_index=True)


def _metrics(df: pd.DataFrame) -> dict[str, float]:
  y = df['patient_label'].astype(int)
  prob = df['patient_prob'].astype(float)
  pred = df['patient_pred'].astype(int)
  return {
    'auc': roc_auc_score(y, prob),
    'f1': f1_score(y, pred),
    'acc': accuracy_score(y, pred),
    'fn': int(((y == 1) & (pred == 0)).sum()),
    'fp': int(((y == 0) & (pred == 1)).sum()),
  }


def _metric_table(all_preds: pd.DataFrame) -> pd.DataFrame:
  rows = []
  for (method, run), group in all_preds.groupby(['method', 'run']):
    row = {'method': method, 'run': run}
    row.update(_metrics(group))
    rows.append(row)
  per_run = pd.DataFrame(rows)
  summary = per_run.groupby('method').agg(
    auc_mean=('auc', 'mean'),
    auc_std=('auc', 'std'),
    f1_mean=('f1', 'mean'),
    f1_std=('f1', 'std'),
    acc_mean=('acc', 'mean'),
    fn_mean=('fn', 'mean'),
    fp_mean=('fp', 'mean'),
  )
  return summary.reset_index()


def _teacher_frame() -> pd.DataFrame:
  frames = []
  for run_idx in range(5):
    df = _read_predictions(TEACHER_ROOT, run_idx)
    df = df.rename(columns={
      'patient_prob': 'teacher_prob',
      'patient_pred': 'teacher_pred',
    })
    df['teacher_confidence'] = (df['teacher_prob'] - 0.5).abs() * 2.0
    df['teacher_correct'] = df['teacher_pred'].astype(int) == df['patient_label'].astype(int)
    frames.append(df[['run', 'patient_id', 'teacher_prob', 'teacher_pred', 'teacher_confidence', 'teacher_correct']])
  return pd.concat(frames, ignore_index=True)


def _paired(base: pd.DataFrame, other: pd.DataFrame, teacher: pd.DataFrame) -> pd.DataFrame:
  merged = base.merge(
    other,
    on=['run', 'patient_id', 'patient_label'],
    suffixes=('_base', '_other'),
  ).merge(teacher, on=['run', 'patient_id'], how='left')
  y = merged['patient_label'].astype(int)
  base_correct = merged['patient_pred_base'].astype(int) == y
  other_correct = merged['patient_pred_other'].astype(int) == y
  merged['base_correct'] = base_correct
  merged['other_correct'] = other_correct
  merged['fixed'] = (~base_correct) & other_correct
  merged['regressed'] = base_correct & (~other_correct)
  merged['both_correct'] = base_correct & other_correct
  merged['both_wrong'] = (~base_correct) & (~other_correct)
  return merged


def _flip_table(paired: pd.DataFrame, name: str) -> dict[str, object]:
  fixed = int(paired['fixed'].sum())
  regressed = int(paired['regressed'].sum())
  both_wrong = int(paired['both_wrong'].sum())
  both_correct = int(paired['both_correct'].sum())
  rows = {
    'comparison': name,
    'fixed': fixed,
    'regressed': regressed,
    'net_fixed': fixed - regressed,
    'both_wrong': both_wrong,
    'both_correct': both_correct,
  }
  for label, label_name in [(1, 'positive'), (0, 'negative')]:
    part = paired[paired['patient_label'] == label]
    rows[f'{label_name}_fixed'] = int(part['fixed'].sum())
    rows[f'{label_name}_regressed'] = int(part['regressed'].sum())
  return rows


def _teacher_group_table(paired: pd.DataFrame, name: str) -> pd.DataFrame:
  bins = [-0.01, 0.5, 0.8, 1.01]
  labels = ['low<=0.5', 'mid(0.5,0.8]', 'high>0.8']
  df = paired.copy()
  df['teacher_conf_bin'] = pd.cut(df['teacher_confidence'], bins=bins, labels=labels)
  rows = []
  for (conf_bin, teacher_correct), group in df.groupby(['teacher_conf_bin', 'teacher_correct'], observed=True):
    if group.empty:
      continue
    rows.append({
      'comparison': name,
      'teacher_conf_bin': str(conf_bin),
      'teacher_correct': bool(teacher_correct),
      'n': len(group),
      'base_acc': float(group['base_correct'].mean()),
      'other_acc': float(group['other_correct'].mean()),
      'delta_acc': float(group['other_correct'].mean() - group['base_correct'].mean()),
      'fixed': int(group['fixed'].sum()),
      'regressed': int(group['regressed'].sum()),
    })
  return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, float_cols: set[str] | None = None) -> str:
  float_cols = float_cols or set()
  out = df.copy()
  for col in float_cols:
    if col in out:
      out[col] = out[col].map(lambda x: f'{x:.4f}')
  out = out.fillna('')
  headers = [str(col) for col in out.columns]
  rows = [[str(value) for value in row] for row in out.to_numpy()]
  widths = [
    max([len(headers[idx])] + [len(row[idx]) for row in rows])
    for idx in range(len(headers))
  ]

  def fmt_row(values: list[str]) -> str:
    return '| ' + ' | '.join(
      value.ljust(widths[idx]) for idx, value in enumerate(values)
    ) + ' |'

  separator = '| ' + ' | '.join('-' * width for width in widths) + ' |'
  return '\n'.join([fmt_row(headers), separator] + [fmt_row(row) for row in rows])


def main() -> None:
  all_preds = pd.concat([_read_method(spec) for spec in METHODS], ignore_index=True)
  metrics = _metric_table(all_preds)
  teacher = _teacher_frame()
  base = all_preds[all_preds['method'] == 'RKD'].drop(columns='method')

  flip_rows = []
  group_tables = []
  for method in ['RKD+TGA_0.5', 'RKD+TGA_0.75']:
    other = all_preds[all_preds['method'] == method].drop(columns='method')
    paired = _paired(base, other, teacher)
    flip_rows.append(_flip_table(paired, f'RKD vs {method}'))
    group_tables.append(_teacher_group_table(paired, f'RKD vs {method}'))

  flips = pd.DataFrame(flip_rows)
  teacher_groups = pd.concat(group_tables, ignore_index=True)

  lines = [
    '# RKD vs RKD+TGA Diagnostic',
    '',
    '## Scope',
    '- Uses saved patient-level `cv_predictions.csv`; no retraining.',
    '- Compares RKD against RKD+TGA weights 0.5 and 0.75.',
    '- Treats each run-patient pair as a diagnostic sample; this is not an independence claim.',
    '- Reports pooled patient-level metrics from saved predictions; these are diagnostic and do not replace logged fold-level metrics.',
    '- Caveat: historical RKD output uses the old teacher alias in logs, while RKD+TGA uses `run_concat_HE_CD20_CD3_mlp_bs32_lr4em4`.',
    '',
    '## Per-Run Summary',
    _markdown_table(
      metrics,
      {'auc_mean', 'auc_std', 'f1_mean', 'f1_std', 'acc_mean', 'fn_mean', 'fp_mean'},
    ),
    '',
    '## Paired Error Flips',
    _markdown_table(flips),
    '',
    '## Teacher Confidence Groups',
    _markdown_table(
      teacher_groups,
      {'base_acc', 'other_acc', 'delta_acc'},
    ),
    '',
    '## Read',
    '- A useful TGA signal should show more fixed cases than regressed cases, especially in teacher-correct/high-confidence groups.',
    '- If fixed and regressed counts are close, the aggregate F1 gain is likely noise-level rather than a reliable mechanism.',
  ]
  REPORT_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
  print(REPORT_PATH)


if __name__ == '__main__':
  main()
