"""Teacher manifest 读取工具。

蒸馏脚本通过 load_manifest() 加载 teacher 训练产生的 manifest.json，
自动获取 fold 参数、模态路径、checkpoint 模板等，消除手动参数对齐。

用法:
  manifest = load_manifest('runs/outputs/concat_HE_CD20_mlp/manifest.json')
  manifest.n_runs          # 5
  manifest.slide_modality_paths  # {"HE": "/.../HE", "CD20": "/.../CD20"}
  manifest.ckpt_tmpl       # 绝对路径模板
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TeacherManifest:
  """从 teacher manifest.json 解析出的结构化信息。"""
  condition_name: str
  n_runs: int
  k_folds: int
  base_seed: int
  modality_names: List[str]            # 模态顺序（决定 slide_concat 拼接顺序）
  slide_modality_paths: Dict[str, str] # 模态名 → slide 特征目录
  ckpt_tmpl: str                       # 已转为绝对路径的 checkpoint 模板


def load_manifest(manifest_path: str) -> TeacherManifest:
  """读取 teacher manifest，返回 TeacherManifest。

  manifest 不存在时抛出 FileNotFoundError，提示先运行 PathoML teacher 训练。

  Args:
    manifest_path: teacher manifest.json 的路径。
  """
  if not os.path.isfile(manifest_path):
    raise FileNotFoundError(
      f"Teacher manifest 不存在: {manifest_path}\n"
      f"请先运行对应的 PathoML teacher 训练脚本以生成 manifest。"
    )

  with open(manifest_path, "r", encoding="utf-8") as f:
    data = json.load(f)

  # (1) 将相对 ckpt_template 转为绝对路径模板
  manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
  ckpt_tmpl = os.path.join(manifest_dir, data["ckpt_template"])

  manifest = TeacherManifest(
    condition_name=data["condition_name"],
    n_runs=data["n_runs"],
    k_folds=data["k_folds"],
    base_seed=data["base_seed"],
    modality_names=data.get("modality_names", []),
    slide_modality_paths=data.get("modality_paths", {}),
    ckpt_tmpl=ckpt_tmpl,
  )

  # (2) 打印摘要，方便维护者确认
  modalities = ", ".join(manifest.modality_names) or "N/A"
  print(
    f"Teacher manifest loaded: {manifest.condition_name}\n"
    f"  n_runs={manifest.n_runs}, k_folds={manifest.k_folds}, "
    f"base_seed={manifest.base_seed}\n"
    f"  modalities: {modalities}"
  )

  return manifest
