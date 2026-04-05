# 只跑 HE+CD20+CD3 MLP bs32 条件，然后从 cv_predictions.csv 统计疑难 case。
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
  run_condition, find_common_sample_keys,
  RunTimeConfig,
  SLIDE_FEAT_ROOT, LABELS_CSV,
  N_RUNS, K_FOLDS, EPOCHS, WD, DROPOUT_RATE, SLIDE_LR, PATIENCE,
  OUTPUTS_DIR, BASE_SEED,
  load_all_module, create_dataset_from_config,
)

DEVICE = "cuda:2"
MLP_HIDDEN_DIM = 128
STAINS = ["HE", "CD20", "CD3"]
CONDITION_NAME = "run_concat_HE_CD20_CD3_mlp__bs32"


def make_config(common_keys) -> RunTimeConfig:
  config = RunTimeConfig()
  config.dataset.dataset_name = "MultimodalConcatSlideDataset"
  config.dataset.dataset_kwargs = {
    "data_root": SLIDE_FEAT_ROOT,
    "modality_names": STAINS,
    "allowed_sample_keys": common_keys,
    "labels_csv": LABELS_CSV,
  }
  config.model.model_name = "mlp"
  config.model.model_kwargs = {"hidden_dim": MLP_HIDDEN_DIM, "dropout": DROPOUT_RATE, "num_layers": 1}
  config.training.device = DEVICE
  config.training.epochs = EPOCHS
  config.training.patience = PATIENCE
  config.training.learning_rate = SLIDE_LR
  config.training.weight_decay = WD
  config.training.batch_size = 32
  return config


def analyze_hard_cases(output_dir: str, n_runs: int, common_keys):
  """从各 run 的 cv_predictions.csv 收集 patient-level 预测，统计疑难 case。"""
  # (1) 收集所有 run 的 patient-level 预测
  run_dfs = []
  for i in range(n_runs):
    csv_path = os.path.join(output_dir, CONDITION_NAME, f"run_{i:02d}", "cv_predictions.csv")
    if not os.path.exists(csv_path):
      print(f"WARNING: {csv_path} not found, skipping")
      continue
    df = pd.read_csv(csv_path)
    # 去重到 patient 级别（CSV 中每个 patient 可能有多个 slide 行）
    patient_df = df.groupby("patient_id").first().reset_index()
    patient_df["run"] = i
    run_dfs.append(patient_df[["patient_id", "patient_label", "patient_prob", "patient_pred", "run"]])

  if not run_dfs:
    print("No prediction CSVs found!")
    return

  all_preds = pd.concat(run_dfs, ignore_index=True)

  # (2) 按 patient 聚合：统计被误判的次数
  patients = all_preds.groupby("patient_id").agg(
    true_label=("patient_label", "first"),
    mean_prob=("patient_prob", "mean"),
    std_prob=("patient_prob", "std"),
    wrong_count=("patient_pred", lambda x: int((x != all_preds.loc[x.index, "patient_label"]).sum())),
    total_runs=("run", "count"),
  ).reset_index()

  # (3) 添加每次 run 的概率
  for i in range(n_runs):
    run_data = all_preds[all_preds["run"] == i][["patient_id", "patient_prob"]]
    patients = patients.merge(
      run_data.rename(columns={"patient_prob": f"run{i}_prob"}),
      on="patient_id", how="left",
    )

  # (4) 筛选疑难 case（至少 1 次被误判）
  hard = patients[patients["wrong_count"] >= 1].sort_values("wrong_count", ascending=False)

  # (5) 添加 slide_ids 列
  cfg = make_config(common_keys)
  load_all_module(cfg)
  ds = create_dataset_from_config(cfg.dataset)
  slide_map = {}
  for idx in range(len(ds)):
    item = ds[idx]
    pid = item["patient_id"]
    sid = item["slide_id"]
    slide_map.setdefault(pid, set()).add(sid)
  slide_map = {pid: ";".join(sorted(sids)) for pid, sids in slide_map.items()}
  hard["slide_ids"] = hard["patient_id"].map(slide_map)

  # (6) 标注真实标签和误判类型
  hard["true_label"] = hard["true_label"].map({0: "Reactive", 1: "MALT"})
  hard["predicted_as"] = hard["true_label"].map({"MALT": "Reactive", "Reactive": "MALT"})

  # (7) 输出
  cols = ["patient_id", "slide_ids", "true_label", "predicted_as", "wrong_count",
          "mean_prob", "std_prob"] + [f"run{i}_prob" for i in range(n_runs)]
  hard = hard[cols]

  out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hard_cases_for_review.csv")
  hard.to_csv(out_path, index=False)
  print(f"\n疑难 case 共 {len(hard)} 个（至少 1/5 次误判），已导出: {out_path}")
  print(f"其中 5/5 次全错: {(hard['wrong_count'] == 5).sum()} 个")
  print(f"其中 ≥3/5 次错: {(hard['wrong_count'] >= 3).sum()} 个")
  print("\n前 20 个疑难 case:")
  print(hard.head(20).to_string(index=False))


def main():
  common_keys = find_common_sample_keys(SLIDE_FEAT_ROOT, STAINS)
  print(f"公共样本数（{' ∩ '.join(STAINS)}）: {len(common_keys)}")

  config = make_config(common_keys)
  results = run_condition(CONDITION_NAME, config, N_RUNS, K_FOLDS, output_dir=OUTPUTS_DIR)

  print(f"\n{'='*60}")
  rm = np.array(results["run_means"])
  ff = np.array(results["all_fold_f1s"])
  print(f"AUC: {rm.mean():.4f} ± {rm.std():.4f}")
  print(f"F1:  {ff.mean():.4f} ± {ff.std():.4f}")
  print(f"{'='*60}")

  # 训练完成，分析疑难 case
  analyze_hard_cases(OUTPUTS_DIR, N_RUNS, common_keys)


if __name__ == "__main__":
  main()
