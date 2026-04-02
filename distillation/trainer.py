"""DistillCrossValidator: 继承 CrossValidator，仅重写 execute 和 _train_epoch。

复用 CrossValidator 的所有 CV 逻辑（折分割、单折流程、评估、CSV导出、汇总打印）。
唯独重写:
  execute()      — 每折前加载对应 fold 的 teacher checkpoint
  _train_epoch() — 双路前向 + 蒸馏损失（由 DistillationLoss 实例决定）
"""

from __future__ import annotations

import os
import time
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from PathoML.config.config import RunTimeConfig
from PathoML.interfaces import BaseDataset
from PathoML.optimization.TrainingStrategy import CrossValidator
from PathoML.optimization.training_utils import TrainingResult, move_to_device, model_inputs
from models.teacher import TeacherMLP
from losses import DistillationLoss


# ---------------------------------------------------------------------------
# DistillCrossValidator
# ---------------------------------------------------------------------------

class DistillCrossValidator(CrossValidator):
  """继承 CrossValidator，仅重写 execute（fold-specific teacher 加载）
  和 _train_epoch（蒸馏损失）。

  用法:
    cv = DistillCrossValidator(
      student_builder   = lambda: StudentABMIL(...),
      dataset           = dataset,
      config            = RunTimeConfig(...),
      distill_loss      = StandardKDLoss(alpha=1, beta=1, temperature=4),
      teacher_ckpt_tmpl = 'path/model_fold_{fold}_best.pth',
      k_folds           = 5,
    )
    Trainer(cv).fit()
  """

  def __init__(
    self,
    student_builder: Callable[[], nn.Module],
    dataset: BaseDataset,
    config: RunTimeConfig,
    distill_loss: DistillationLoss,
    teacher_ckpt_tmpl: str,
    k_folds: int = 5,
  ) -> None:
    super().__init__(student_builder, dataset, config, k_folds)
    self.teacher_ckpt_tmpl = teacher_ckpt_tmpl
    self.distill_loss = distill_loss
    self.teacher = None   # 每折在 execute() 中赋值

  @property
  def name(self) -> str:
    return "KFold CrossValidation (Distillation)"

  # -- execute：唯一差异是每折前加载 teacher --

  def execute(self) -> TrainingResult:
    """K折 CV 主入口。与 CrossValidator.execute() 相同，仅增加 teacher 加载。"""
    os.makedirs(self.logging_cfg.save_dir, exist_ok=True)
    split_iter, patient_ids = self._prepare_splits()

    fold_results, all_test_details = [], []
    cv_start = time.time()

    for fold, (train_val_ids, test_ids) in enumerate(split_iter):
      print(f"\n{'='*70}\nFold {fold+1} / {self.k_folds}\n{'='*70}")

      # (1) 加载当前折对应的 teacher checkpoint（每折不同）
      self.teacher = TeacherMLP.from_checkpoint(
        self.teacher_ckpt_tmpl.format(fold=fold + 1)
      ).to(self.device)

      # (2) 校验 teacher 与 student 的折划分严格一致
      if self.teacher.test_fold is not None:
        s_train = sorted(set(patient_ids[train_val_ids].tolist()))
        s_test  = sorted(set(patient_ids[test_ids].tolist()))
        assert s_train == self.teacher.train_fold, (
          f"Fold {fold+1}: train_fold 患者集不一致，"
          f"请检查 dataset/seed 是否与 teacher 训练时一致。"
        )
        assert s_test == self.teacher.test_fold, (
          f"Fold {fold+1}: test_fold 患者集不一致，"
          f"请检查 dataset/seed 是否与 teacher 训练时一致。"
        )
      else:
        raise ValueError(f"Fold {fold+1}: teacher checkpoint lacks train_fold/test_fold, cannot verify fold splits.")
      print(f"Fold {fold+1}: Teacher checkpoint loaded, fold splits verified.")


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

  # -- _train_epoch：重写以加入 teacher 前向和蒸馏损失 --

  def _train_epoch(
    self,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
  ):
    """蒸馏单 epoch 训练。model 为 student；teacher 通过 self.teacher 访问（已冻结）。

    损失计算委托给 self.distill_loss（DistillationLoss 实例）。
    返回 (avg_loss, accuracy)，签名与父类保持一致。
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    with tqdm(loader, desc="  Training", unit="batch", leave=False) as pbar:
      for raw_batch in pbar:
        batch  = move_to_device(raw_batch, self.device)
        labels = batch['label']

        # (1) Teacher 前向（无梯度）
        with torch.no_grad():
          t_out = self.teacher(batch['slide_concat'])   # (B, sum_of_dims)

        # (2) Student 前向
        inputs = model_inputs(batch)
        s_out  = model(inputs)

        # (3) 蒸馏损失 + 反向传播
        loss = self.distill_loss(s_out, t_out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        probs  = torch.sigmoid(s_out['logits'].view(-1))
        preds  = (probs > self.training_cfg.patient_threshold).float()
        correct += (preds == labels.view(-1).float()).sum().item()
        total  += bs
        pbar.set_postfix(loss=f"{total_loss/total:.4f}")

    return total_loss / total, correct / total
