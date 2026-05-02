"""Mechanism diagnostics for BasicABMIL distillation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
  accuracy_score,
  average_precision_score,
  brier_score_loss,
  f1_score,
  precision_score,
  recall_score,
  roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / 'distillation' / 'experiments' / 'outputs'
TEACHER_ROOT = (
  ROOT / 'teacher' / 'experiments' / 'outputs'
  / 'run_concat_HE_CD20_CD3_mlp_bs32_lr4em4'
)
REPORT_PATH = (
  ROOT / 'distillation' / 'experiments'
  / 'distillation_mechanism_diagnostic.md'
)


@dataclass(frozen=True)
class MethodSpec:
  name: str
  path: Path
  group: str = 'main'


METHODS = [
  MethodSpec('ABMIL', OUTPUT_ROOT / 'task_only_baseline_task'),
  MethodSpec('Hidden', OUTPUT_ROOT / 'hidden_features_matching_task_hidden'),
  MethodSpec('SPKD', OUTPUT_ROOT / 'similarity_preserving_kd_task_similarity_preserving'),
  MethodSpec('StandardKD', OUTPUT_ROOT / 'standard_kd_task_soft_label_t4p0'),
  MethodSpec('DKD_strict', OUTPUT_ROOT / 'decoupled_kd_task_dkd_t4p0_a1p0_b4p0'),
  MethodSpec('RKD', OUTPUT_ROOT / 'rkd_task_rkd_distance_rkd_angle_2p0'),
  MethodSpec(
    'RKD_TGA_0.5',
    OUTPUT_ROOT / 'rkd_tga_task_rkd_distance_rkd_angle_2p0_attn_cosine_logits_no_detach_0p5',
    'experimental',
  ),
  MethodSpec(
    'RKD_confirm',
    OUTPUT_ROOT / 'rkd_task_rkd_distance_rkd_angle_2p0_confirm_student_seed142',
    'confirmation',
  ),
  MethodSpec(
    'RKD_TGA_0.5_confirm',
    OUTPUT_ROOT / 'rkd_tga_task_rkd_distance_rkd_angle_2p0_attn_cosine_logits_no_detach_0p5_confirm_student_seed142',
    'confirmation',
  ),
]


def _read_predictions(path: Path, run_idx: int) -> pd.DataFrame:
  csv_path = path / f'run_{run_idx:02d}' / 'cv_predictions.csv'
  if not csv_path.is_file():
    raise FileNotFoundError(csv_path)
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
    df['group'] = spec.group
    frames.append(df)
  return pd.concat(frames, ignore_index=True)


def _read_all_methods() -> pd.DataFrame:
  return pd.concat([_read_method(spec) for spec in METHODS], ignore_index=True)


def _teacher_frame() -> pd.DataFrame:
  frames = []
  for run_idx in range(5):
    df = _read_predictions(TEACHER_ROOT, run_idx)
    df = df.rename(columns={
      'patient_prob': 'teacher_prob',
      'patient_pred': 'teacher_pred',
    })
    df['teacher_confidence'] = (df['teacher_prob'] - 0.5).abs() * 2.0
    df['teacher_correct'] = (
      df['teacher_pred'].astype(int) == df['patient_label'].astype(int)
    )
    frames.append(
      df[[
        'run',
        'patient_id',
        'teacher_prob',
        'teacher_pred',
        'teacher_confidence',
        'teacher_correct',
      ]]
    )
  return pd.concat(frames, ignore_index=True)


def _confusion_counts(y: pd.Series, pred: pd.Series) -> dict[str, int]:
  y = y.astype(int)
  pred = pred.astype(int)
  return {
    'tp': int(((y == 1) & (pred == 1)).sum()),
    'tn': int(((y == 0) & (pred == 0)).sum()),
    'fp': int(((y == 0) & (pred == 1)).sum()),
    'fn': int(((y == 1) & (pred == 0)).sum()),
  }


def _safe_div(num: float, den: float) -> float:
  return float(num / den) if den else float('nan')


def _ece(y: pd.Series, prob: pd.Series, n_bins: int = 10) -> float:
  y_np = y.astype(float).to_numpy()
  prob_np = prob.astype(float).to_numpy()
  edges = np.linspace(0.0, 1.0, n_bins + 1)
  total = len(prob_np)
  score = 0.0
  for idx in range(n_bins):
    left, right = edges[idx], edges[idx + 1]
    if idx == n_bins - 1:
      mask = (prob_np >= left) & (prob_np <= right)
    else:
      mask = (prob_np >= left) & (prob_np < right)
    if not mask.any():
      continue
    confidence = prob_np[mask].mean()
    accuracy = y_np[mask].mean()
    score += mask.sum() / total * abs(confidence - accuracy)
  return float(score)


def _metrics(df: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
  y = df['patient_label'].astype(int)
  prob = df['patient_prob'].astype(float)
  pred = (prob >= threshold).astype(int)
  counts = _confusion_counts(y, pred)
  return {
    'auc': roc_auc_score(y, prob),
    'pr_auc': average_precision_score(y, prob),
    'f1': f1_score(y, pred),
    'acc': accuracy_score(y, pred),
    'precision': precision_score(y, pred, zero_division=0),
    'recall': recall_score(y, pred, zero_division=0),
    'specificity': _safe_div(counts['tn'], counts['tn'] + counts['fp']),
    'brier': brier_score_loss(y, prob),
    'ece10': _ece(y, prob, 10),
    **counts,
  }


def _method_summary(all_preds: pd.DataFrame) -> pd.DataFrame:
  rows = []
  for (group, method, run), df in all_preds.groupby(['group', 'method', 'run']):
    row = {'group': group, 'method': method, 'run': run}
    row.update(_metrics(df))
    rows.append(row)
  per_run = pd.DataFrame(rows)
  summary = per_run.groupby(['group', 'method']).agg(
    auc_mean=('auc', 'mean'),
    auc_std=('auc', 'std'),
    pr_auc_mean=('pr_auc', 'mean'),
    f1_mean=('f1', 'mean'),
    f1_std=('f1', 'std'),
    acc_mean=('acc', 'mean'),
    recall_mean=('recall', 'mean'),
    specificity_mean=('specificity', 'mean'),
    fp_mean=('fp', 'mean'),
    fn_mean=('fn', 'mean'),
    brier_mean=('brier', 'mean'),
    ece10_mean=('ece10', 'mean'),
  )
  return summary.reset_index()


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


def _flip_row(paired: pd.DataFrame, comparison: str) -> dict[str, object]:
  row = {
    'comparison': comparison,
    'fixed': int(paired['fixed'].sum()),
    'regressed': int(paired['regressed'].sum()),
    'net_fixed': int(paired['fixed'].sum() - paired['regressed'].sum()),
    'both_wrong': int(paired['both_wrong'].sum()),
    'both_correct': int(paired['both_correct'].sum()),
  }
  for label, name in [(1, 'positive'), (0, 'negative')]:
    part = paired[paired['patient_label'] == label]
    row[f'{name}_fixed'] = int(part['fixed'].sum())
    row[f'{name}_regressed'] = int(part['regressed'].sum())
  return row


def _paired_tables(all_preds: pd.DataFrame, teacher: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  comparisons = [
    ('ABMIL', 'StandardKD'),
    ('ABMIL', 'RKD'),
    ('ABMIL', 'RKD_TGA_0.5'),
    ('StandardKD', 'RKD'),
    ('RKD', 'RKD_TGA_0.5'),
    ('RKD_confirm', 'RKD_TGA_0.5_confirm'),
  ]
  flip_rows = []
  teacher_rows = []
  for base_name, other_name in comparisons:
    base = all_preds[all_preds['method'] == base_name].drop(columns=['method', 'group'])
    other = all_preds[all_preds['method'] == other_name].drop(columns=['method', 'group'])
    paired = _paired(base, other, teacher)
    comparison = f'{base_name} vs {other_name}'
    flip_rows.append(_flip_row(paired, comparison))
    teacher_rows.extend(_teacher_group_rows(paired, comparison))
  return pd.DataFrame(flip_rows), pd.DataFrame(teacher_rows)


def _teacher_group_rows(paired: pd.DataFrame, comparison: str) -> list[dict[str, object]]:
  bins = [-0.01, 0.5, 0.8, 1.01]
  labels = ['low<=0.5', 'mid(0.5,0.8]', 'high>0.8']
  df = paired.copy()
  df['teacher_conf_bin'] = pd.cut(df['teacher_confidence'], bins=bins, labels=labels)
  rows = []
  grouped = df.groupby(['teacher_conf_bin', 'teacher_correct'], observed=True)
  for (conf_bin, teacher_correct), group in grouped:
    if group.empty:
      continue
    rows.append({
      'comparison': comparison,
      'teacher_conf_bin': str(conf_bin),
      'teacher_correct': bool(teacher_correct),
      'n': len(group),
      'base_acc': float(group['base_correct'].mean()),
      'other_acc': float(group['other_correct'].mean()),
      'delta_acc': float(group['other_correct'].mean() - group['base_correct'].mean()),
      'fixed': int(group['fixed'].sum()),
      'regressed': int(group['regressed'].sum()),
    })
  return rows


def _best_f1_threshold(df: pd.DataFrame) -> dict[str, float]:
  y = df['patient_label'].astype(int)
  prob = df['patient_prob'].astype(float)
  candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], prob.to_numpy())))
  best = {'threshold': 0.5, 'f1': -1.0, 'acc': 0.0, 'recall': 0.0, 'specificity': 0.0}
  for threshold in candidates:
    pred = (prob >= threshold).astype(int)
    counts = _confusion_counts(y, pred)
    f1 = f1_score(y, pred)
    if f1 > best['f1']:
      best = {
        'threshold': float(threshold),
        'f1': float(f1),
        'acc': float(accuracy_score(y, pred)),
        'recall': _safe_div(counts['tp'], counts['tp'] + counts['fn']),
        'specificity': _safe_div(counts['tn'], counts['tn'] + counts['fp']),
      }
  return best


def _threshold_table(all_preds: pd.DataFrame) -> pd.DataFrame:
  rows = []
  for (group, method, run), df in all_preds.groupby(['group', 'method', 'run']):
    fixed = _metrics(df, 0.5)
    best = _best_f1_threshold(df)
    rows.append({
      'group': group,
      'method': method,
      'run': run,
      'fixed_f1': fixed['f1'],
      'fixed_recall': fixed['recall'],
      'fixed_specificity': fixed['specificity'],
      'best_threshold': best['threshold'],
      'best_f1': best['f1'],
      'best_recall': best['recall'],
      'best_specificity': best['specificity'],
      'f1_gain': best['f1'] - fixed['f1'],
    })
  per_run = pd.DataFrame(rows)
  return per_run.groupby(['group', 'method']).agg(
    fixed_f1_mean=('fixed_f1', 'mean'),
    best_f1_mean=('best_f1', 'mean'),
    f1_gain_mean=('f1_gain', 'mean'),
    best_threshold_mean=('best_threshold', 'mean'),
    fixed_recall_mean=('fixed_recall', 'mean'),
    best_recall_mean=('best_recall', 'mean'),
    fixed_specificity_mean=('fixed_specificity', 'mean'),
    best_specificity_mean=('best_specificity', 'mean'),
  ).reset_index()


def _calibration_bins(all_preds: pd.DataFrame) -> pd.DataFrame:
  focus = ['ABMIL', 'StandardKD', 'RKD', 'RKD_TGA_0.5']
  df = all_preds[all_preds['method'].isin(focus)].copy()
  bins = np.linspace(0.0, 1.0, 6)
  labels = ['[0,0.2)', '[0.2,0.4)', '[0.4,0.6)', '[0.6,0.8)', '[0.8,1]']
  df['prob_bin'] = pd.cut(df['patient_prob'], bins=bins, labels=labels, include_lowest=True)
  rows = []
  for (method, prob_bin), group in df.groupby(['method', 'prob_bin'], observed=True):
    rows.append({
      'method': method,
      'prob_bin': str(prob_bin),
      'n': len(group),
      'mean_prob': float(group['patient_prob'].mean()),
      'positive_rate': float(group['patient_label'].mean()),
      'abs_gap': float(abs(group['patient_prob'].mean() - group['patient_label'].mean())),
    })
  return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, float_cols: set[str] | None = None) -> str:
  float_cols = float_cols or set()
  out = df.copy()
  for col in float_cols:
    if col in out:
      out[col] = out[col].map(lambda value: f'{float(value):.4f}')
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
  all_preds = _read_all_methods()
  teacher = _teacher_frame()

  summary = _method_summary(all_preds)
  flips, teacher_groups = _paired_tables(all_preds, teacher)
  thresholds = _threshold_table(all_preds)
  calibration = _calibration_bins(all_preds)

  float_cols = {
    'auc_mean',
    'auc_std',
    'pr_auc_mean',
    'f1_mean',
    'f1_std',
    'acc_mean',
    'recall_mean',
    'specificity_mean',
    'fp_mean',
    'fn_mean',
    'brier_mean',
    'ece10_mean',
    'base_acc',
    'other_acc',
    'delta_acc',
    'fixed_f1_mean',
    'best_f1_mean',
    'f1_gain_mean',
    'best_threshold_mean',
    'fixed_recall_mean',
    'best_recall_mean',
    'fixed_specificity_mean',
    'best_specificity_mean',
    'mean_prob',
    'positive_rate',
    'abs_gap',
  }

  lines = [
    '# Distillation Mechanism Diagnostic',
    '',
    '## Scope',
    '- Uses saved patient-level `cv_predictions.csv`; no retraining.',
    '- Uses the fixed `StudentBasicABMIL` platform only.',
    '- Reports pooled patient-level diagnostics from saved predictions.',
    '- Logged fold-level metrics in `PLAN.md` remain the primary experiment results.',
    '- Threshold sweeps are diagnostic upper bounds on saved test predictions; they are not deployable validation-tuned thresholds.',
    '',
    '## Method Summary',
    _markdown_table(summary, float_cols),
    '',
    '## Paired Error Flips',
    _markdown_table(flips),
    '',
    '## Teacher Confidence Groups',
    _markdown_table(teacher_groups, float_cols),
    '',
    '## Threshold Sweep Diagnostic',
    _markdown_table(thresholds, float_cols),
    '',
    '## Calibration Bins',
    _markdown_table(calibration, float_cols),
    '',
    '## Read',
    '- RKD is the strongest fixed-threshold F1 mechanism on the main BasicABMIL line in this diagnostic: it reduces mean FN versus ABMIL (`19.6 -> 14.6`) while leaving mean FP nearly unchanged (`28.6 -> 28.4`).',
    '- RKD improves error flips over ABMIL more than StandardKD does (`net_fixed=26` vs `15`), and still improves over StandardKD directly (`net_fixed=11`).',
    '- RKD gains are not a clean calibration win: Brier/ECE are worse than ABMIL and StandardKD, so the useful effect is better interpreted as recall/FN operating-point behavior.',
    '- RKD is also not a clear pooled ranking winner in this saved-prediction diagnostic; logged fold-level metrics remain the primary result for AUC comparisons.',
    '- RKD+TGA 0.5 remains experimental: it improves F1/recall and confirmation paired flips, but the main RKD-vs-TGA paired flip result is near cancelled (`net_fixed=1`) and the confirmation AUC does not beat RKD.',
    '- Threshold sweeps do not erase RKD: tuned diagnostic F1 remains highest for RKD+TGA 0.5 experimental (`0.8636`) and RKD remains the best main-table method (`0.8595`).',
    '- Next mechanism work should focus on RKD-style relation/operating-point analysis, not new TGA variants or stronger students.',
  ]
  REPORT_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
  print(REPORT_PATH)


if __name__ == '__main__':
  main()
