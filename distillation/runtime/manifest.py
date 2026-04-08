"""Teacher artifact manifest loader for distillation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List


@dataclass
class TeacherManifest:
  """Structured teacher artifact contract consumed by distillation."""

  schema_version: int
  artifact_type: str
  producer_system: str
  condition_name: str
  n_runs: int
  k_folds: int
  base_seed: int
  modality_names: List[str]
  data_root: str
  labels_csv: str
  model_name: str
  model_kwargs: dict
  sample_set_fingerprint: str
  ckpt_template: str

  @property
  def ckpt_tmpl(self) -> str:
    return self.ckpt_template


def load_manifest(manifest_path: str) -> TeacherManifest:
  """Load teacher manifest.json and normalize it into TeacherManifest."""
  if not os.path.isfile(manifest_path):
    raise FileNotFoundError(
      f"Teacher manifest not found: {manifest_path}\n"
      "Please run the corresponding teacher experiment first."
    )

  with open(manifest_path, "r", encoding="utf-8") as f:
    data = json.load(f)

  manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
  ckpt_template = os.path.join(manifest_dir, data["ckpt_template"])

  manifest = TeacherManifest(
    schema_version=int(data.get("schema_version", 0)),
    artifact_type=data.get("artifact_type", "teacher_model"),
    producer_system=data.get("producer_system", "teacher"),
    condition_name=data["condition_name"],
    n_runs=data["n_runs"],
    k_folds=data["k_folds"],
    base_seed=data["base_seed"],
    modality_names=data.get("modality_names", []),
    data_root=data.get("data_root") or (
      os.path.dirname(next(iter(data["modality_paths"].values())))
      if "modality_paths" in data else ""
    ),
    labels_csv=data.get("labels_csv", ""),
    model_name=data.get("model_name", ""),
    model_kwargs=dict(data.get("model_kwargs", {})),
    sample_set_fingerprint=data.get("sample_set_fingerprint", ""),
    ckpt_template=ckpt_template,
  )

  modalities = ", ".join(manifest.modality_names) or "N/A"
  print(
    f"Teacher manifest loaded: {manifest.condition_name}\n"
    f"  n_runs={manifest.n_runs}, k_folds={manifest.k_folds}, "
    f"base_seed={manifest.base_seed}\n"
    f"  modalities: {modalities}\n"
    f"  schema_version={manifest.schema_version}, producer={manifest.producer_system}"
  )
  return manifest
