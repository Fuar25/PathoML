"""Distillation runtime adapter built on top of PathoML cross-validation."""

from __future__ import annotations

import os
import time
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from PathoML.config.config import RunTimeConfig
from PathoML.dataset.utils import _variable_size_collate
from PathoML.interfaces import BaseDataset
from PathoML.optimization.TrainingStrategy import CrossValidator
from PathoML.optimization.training_utils import TrainingResult, move_to_device, model_inputs
from distillation.losses import DistillationLoss
from distillation.models.teacher import TeacherMLP
from distillation.models.teacher import RegistryTeacher


class DistillCrossValidator(CrossValidator):
  """Cross-validator with teacher checkpoint loading and distillation loss injection."""

  def __init__(
    self,
    student_builder: Callable[[], nn.Module],
    dataset: BaseDataset,
    config: RunTimeConfig,
    distill_loss: DistillationLoss,
    teacher_ckpt_tmpl: str,
    teacher_manifest=None,
    k_folds: int = 5,
    cache_teacher_outputs: bool = True,
    teacher_output_cache_batch_size: int = 64,
  ) -> None:
    super().__init__(student_builder, dataset, config, k_folds)
    self.teacher_ckpt_tmpl = teacher_ckpt_tmpl
    self.teacher_manifest = teacher_manifest
    self.distill_loss = distill_loss
    self.teacher = None
    self.cache_teacher_outputs = bool(cache_teacher_outputs)
    self.teacher_output_cache_batch_size = int(teacher_output_cache_batch_size)
    self.teacher_output_cache: dict[str, torch.Tensor] | None = None
    self.teacher_input_dataset = None
    self.teacher_input_index_by_key: dict[tuple[str, str], int] = {}

  @property
  def name(self) -> str:
    return "KFold CrossValidation (Distillation)"

  def execute(self) -> TrainingResult:
    """Run K-fold CV while validating the teacher checkpoint for each fold."""
    os.makedirs(self.logging_cfg.save_dir, exist_ok=True)
    split_iter, patient_ids = self._prepare_splits()

    fold_results, all_test_details = [], []
    cv_start = time.time()

    for fold, (train_val_ids, test_ids) in enumerate(split_iter):
      print(f"\n{'='*70}\nFold {fold+1} / {self.k_folds}\n{'='*70}")
      self.teacher = self._load_teacher_checkpoint(
        self.teacher_ckpt_tmpl.format(fold=fold + 1)
      ).to(self.device)

      if self.teacher.test_fold is not None:
        s_train = sorted(set(patient_ids[train_val_ids].tolist()))
        s_test = sorted(set(patient_ids[test_ids].tolist()))
        assert s_train == self.teacher.train_fold, (
          f"Fold {fold+1}: train_fold mismatch between teacher and distillation dataset."
        )
        assert s_test == self.teacher.test_fold, (
          f"Fold {fold+1}: test_fold mismatch between teacher and distillation dataset."
        )
      else:
        raise ValueError(
          f"Fold {fold+1}: teacher checkpoint lacks train_fold/test_fold and cannot be verified."
        )
      print(f"Fold {fold+1}: Teacher checkpoint loaded, fold splits verified.")
      if self.cache_teacher_outputs:
        self.teacher_output_cache = self._precompute_teacher_outputs()
      else:
        if not self._uses_slide_teacher_inputs:
          raise ValueError(
            "Registry-backed teachers require cached teacher outputs. "
            "Set PATHOML_DISTILLATION_CACHE_TEACHER_OUTPUTS=1."
          )
        self.teacher_output_cache = None

      result, test_details = self._train_single_fold(
        fold=fold + 1,
        train_val_ids=train_val_ids,
        test_ids=test_ids,
        patient_ids=patient_ids,
      )
      fold_results.append(result)
      all_test_details.append(test_details)

    self._print_cv_summary(fold_results, time.time() - cv_start)
    self._save_cv_predictions(all_test_details)

    return TrainingResult(
      strategy_name=self.name,
      fold_results=fold_results,
      result_dir=self.logging_cfg.save_dir,
    )

  @property
  def _uses_slide_teacher_inputs(self) -> bool:
    model_name = getattr(self.teacher_manifest, 'model_name', '') if self.teacher_manifest else ''
    return model_name in {'', 'mlp'}

  def _load_teacher_checkpoint(self, ckpt_path: str) -> nn.Module:
    if self._uses_slide_teacher_inputs:
      return TeacherMLP.from_checkpoint(ckpt_path)
    teacher_dataset = self._ensure_teacher_input_dataset()
    return RegistryTeacher.from_manifest_checkpoint(
      self.teacher_manifest,
      ckpt_path,
      teacher_dataset,
    )

  def _ensure_teacher_input_dataset(self):
    if self.teacher_input_dataset is not None:
      return self.teacher_input_dataset
    if not str(self.teacher_manifest.model_name).startswith('registered_patch_'):
      raise ValueError(
        "Only legacy MLP and registered patch teacher manifests are supported. "
        f"Got model_name={self.teacher_manifest.model_name!r}."
      )
    if not hasattr(self.dataset, 'get_sample_keys'):
      raise ValueError("Registry-backed teacher loading requires dataset.get_sample_keys().")

    from teacher.dataset.registered_multimodal_patch import RegisteredMultimodalPatchDataset

    allowed_keys = set(self.dataset.get_sample_keys())
    teacher_dataset = RegisteredMultimodalPatchDataset(
      data_root=self.teacher_manifest.data_root,
      modality_names=list(self.teacher_manifest.modality_names),
      labels_csv=self.teacher_manifest.labels_csv,
      min_aligned_patches=1,
      alignment_mode='union',
      cache_aligned=True,
      verbose=False,
      allowed_sample_keys=allowed_keys,
    )
    self.teacher_input_index_by_key = {
      item['sample_key']: idx
      for idx, item in enumerate(teacher_dataset.samples)
    }
    missing = sorted(allowed_keys - set(self.teacher_input_index_by_key))
    if missing:
      raise ValueError(
        "Teacher input dataset is missing samples required by distillation: "
        f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
      )
    self.teacher_input_dataset = teacher_dataset
    return teacher_dataset

  def _precompute_teacher_outputs(self) -> dict[str, torch.Tensor]:
    """Precompute fold-local teacher outputs for every dataset sample."""
    if not self._uses_slide_teacher_inputs:
      return self._precompute_registry_teacher_outputs()

    if not hasattr(self.dataset, 'get_slide_concat'):
      raise ValueError(
        "Teacher output cache requires dataset.get_slide_concat(idx). "
        "Disable PATHOML_DISTILLATION_CACHE_TEACHER_OUTPUTS for this dataset."
      )

    batch_size = max(1, self.teacher_output_cache_batch_size)
    hidden_chunks, logit_chunks = [], []
    class_weight = None

    self.teacher.eval()
    with torch.no_grad():
      for start in range(0, len(self.dataset), batch_size):
        stop = min(start + batch_size, len(self.dataset))
        slide_batch = torch.stack([
          self.dataset.get_slide_concat(idx)
          for idx in range(start, stop)
        ]).to(self.device)
        out = self.teacher(slide_batch)
        hidden_chunks.append(out['hidden'].detach().cpu())
        logit_chunks.append(out['logit'].detach().cpu())
        if class_weight is None and 'class_weight' in out:
          class_weight = out['class_weight'].detach().cpu()

    if class_weight is None:
      raise ValueError("Teacher output cache requires teacher forward to return class_weight.")

    cache = {
      'hidden': torch.cat(hidden_chunks, dim=0),
      'logit': torch.cat(logit_chunks, dim=0),
      'class_weight': class_weight,
    }
    print(
      "Fold teacher outputs cached: "
      f"{cache['hidden'].shape[0]} samples, hidden_dim={cache['hidden'].shape[1]}"
    )
    return cache

  def _precompute_registry_teacher_outputs(self) -> dict[str, torch.Tensor]:
    """Precompute outputs from a registry-backed teacher dataset in distillation order."""
    teacher_dataset = self._ensure_teacher_input_dataset()
    sample_keys = self.dataset.get_sample_keys()
    batch_size = max(1, self.teacher_output_cache_batch_size)
    hidden_chunks, logit_chunks = [], []
    class_weight = None

    self.teacher.eval()
    with torch.no_grad():
      for start in range(0, len(sample_keys), batch_size):
        stop = min(start + batch_size, len(sample_keys))
        items = [
          teacher_dataset[self.teacher_input_index_by_key[sample_keys[idx]]]
          for idx in range(start, stop)
        ]
        batch = move_to_device(_variable_size_collate(items), self.device)
        out = self.teacher(batch)
        hidden_chunks.append(out['hidden'].detach().cpu())
        logit_chunks.append(out['logit'].detach().cpu())
        if class_weight is None and 'class_weight' in out:
          class_weight = out['class_weight'].detach().cpu()

    cache = {
      'hidden': torch.cat(hidden_chunks, dim=0),
      'logit': torch.cat(logit_chunks, dim=0),
    }
    if class_weight is not None:
      cache['class_weight'] = class_weight
    print(
      "Fold registry teacher outputs cached: "
      f"{cache['hidden'].shape[0]} samples, hidden_dim={cache['hidden'].shape[1]}"
    )
    return cache

  def _cached_teacher_outputs_for_batch(self, batch: dict) -> dict:
    if self.teacher_output_cache is None:
      raise RuntimeError("Teacher output cache is not initialized.")
    if 'sample_index' not in batch:
      raise ValueError(
        "Teacher output cache requires `sample_index` in the batch. "
        "Disable PATHOML_DISTILLATION_CACHE_TEACHER_OUTPUTS for this dataset."
      )

    indices = batch['sample_index'].detach().cpu().long()
    return {
      'hidden': self.teacher_output_cache['hidden'][indices].to(self.device),
      'logit': self.teacher_output_cache['logit'][indices].to(self.device),
      **(
        {'class_weight': self.teacher_output_cache['class_weight'].to(self.device)}
        if 'class_weight' in self.teacher_output_cache else {}
      ),
    }

  def _train_epoch(
    self,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
  ):
    """Train one distillation epoch with frozen teacher forward passes."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    with tqdm(loader, desc="  Training", unit="batch", leave=False) as pbar:
      for raw_batch in pbar:
        batch = move_to_device(raw_batch, self.device)
        labels = batch['label']

        if self.teacher_output_cache is not None:
          t_out = self._cached_teacher_outputs_for_batch(batch)
        else:
          if not self._uses_slide_teacher_inputs:
            raise ValueError(
              "Registry-backed teachers require cached teacher outputs. "
              "Set PATHOML_DISTILLATION_CACHE_TEACHER_OUTPUTS=1."
            )
          with torch.no_grad():
            t_out = self.teacher(batch['slide_concat'])

        inputs = model_inputs(batch)
        s_out = model(inputs)

        loss = self.distill_loss(s_out, t_out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        probs = torch.sigmoid(s_out['logits'].view(-1))
        preds = (probs > self.training_cfg.patient_threshold).float()
        correct += (preds == labels.view(-1).float()).sum().item()
        total += batch_size
        pbar.set_postfix(loss=f"{total_loss/total:.4f}")

    return total_loss / total, correct / total
