"""基于 Optuna 的超参数优化模块（实验性功能）。

此脚本已从核心库移出，作为可选的实验工具使用。
"""

# 标准库
import copy
import csv
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 第三方库
import numpy as np
import optuna
import torch

# 项目内部模块
from core.trainer import Trainer


class HyperParameterOptimizer:
    """基于贝叶斯优化（Optuna）的超参数搜索器。

    通过在指定搜索空间内运行多组超参数组合的交叉验证，
    找到使平均测试 AUC 最高的超参数配置。

    Attributes:
        model_builder: 可调用对象，每次调用返回一个新的模型实例。
        dataset: 完整数据集对象。
        base_config: 基础运行配置（RunTimeConfig），搜索过程中会对其深拷贝后修改。
    """

    def __init__(self, model_builder, dataset, base_config) -> None:
        """初始化超参数优化器。

        Args:
            model_builder: 可调用对象，每次调用返回一个未初始化的模型实例。
            dataset: 完整数据集对象，需兼容 BaseDataset 接口。
            base_config: 基础运行配置（RunTimeConfig），搜索过程中按 trial 深拷贝并修改。
        """
        self.model_builder = model_builder
        self.dataset = dataset
        self.base_config = base_config

    def objective(self, trial: optuna.Trial) -> float:
        """单次 Optuna trial 的目标函数。

        深拷贝基础配置后，对学习率、权重衰减、Dropout、隐藏维度和注意力头数
        进行采样，随后运行交叉验证并以平均测试 AUC 作为优化目标。
        每次 trial 的结果会追加写入 CSV 日志文件以供后续分析。

        Args:
            trial: Optuna trial 对象，用于从搜索空间中采样超参数。

        Returns:
            本次 trial 各折的平均测试 AUC。
        """
        # 深拷贝配置，避免不同 trial 间相互污染
        config = copy.deepcopy(self.base_config)

        # 固定随机种子，确保相同超参数每次产生相同结果，以便公平比较
        seed = config.training.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # 定义超参数搜索空间
        config.training.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        config.training.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        config.model.dropout = trial.suggest_float("dropout", 0, 0.5)
        config.model.hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768, 1024])
        config.model.n_heads = trial.suggest_categorical("n_heads", [1, 4, 8])


        # 为每个 trial 设置独立的保存目录，防止模型文件互相覆盖
        config.logging.save_dir = f"{self.base_config.logging.save_dir}/trial_{trial.number}"
        # 启用结果日志以计算 patient 级别的指标（必要用于评估），但关闭 CSV 保存以节省存储
        config.logging.log_test_results = True  # 保持启用，以计算 patient 级别指标

        # 运行交叉验证
        trainer = Trainer(self.model_builder, self.dataset, config)
        fold_results = trainer.cross_validate(k_folds=config.training.k_folds)

        # 计算各折平均指标
        avg_test_auc = np.mean([r['test_auc'] for r in fold_results])
        avg_test_acc = np.mean([r['test_acc'] for r in fold_results])

        # 将本次 trial 结果追加写入 CSV 日志
        log_path = os.path.join(self.base_config.logging.save_dir, "hpo_log.csv")
        os.makedirs(self.base_config.logging.save_dir, exist_ok=True)

        log_data = {
            "trial": trial.number,
            "avg_test_auc": avg_test_auc,
            "avg_test_acc": avg_test_acc,
        }
        log_data.update(trial.params)  # 记录原始搜索值
        
        file_exists = os.path.isfile(log_path)
        with open(log_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)

        return avg_test_auc

    def optimize(self, n_trials: int = 20, study_name: str = "abmil_optimization") -> optuna.Trial:
        """启动超参数搜索，返回最优 trial。

        Args:
            n_trials: 搜索的 trial 总次数。
            study_name: Optuna study 的名称标识。

        Returns:
            包含最优超参数配置的 Optuna Trial 对象。
        """
        print(f"Starting Bayesian Optimization with {n_trials} trials...")

        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.optimize(self.objective, n_trials=n_trials)

        print("\nOptimization Complete!")
        print("=" * 70)
        best_trial = study.best_trial
        print(f"Best Trial #{best_trial.number} -> Avg Test AUC: {best_trial.value:.4f}")
        print("=" * 70)
        
        print("\n⚠️  Note: trial.params contains raw search values (not normalized).")
        print("    See 'Best Fusion Weights' below for actual normalized weights used in training.\n")
        
        # 提取并分类显示超参数
        model_params = {}
        training_params = {}
        modality_weights = {}
        
        for key, value in best_trial.params.items():
            if key.startswith("weight_"):
                modality = key.replace("weight_", "")
                modality_weights[modality] = value
            elif key in ["learning_rate", "weight_decay"]:
                training_params[key] = value
            else:
                model_params[key] = value
        
        # 打印模型参数
        if model_params:
            print("\nBest Model Parameters:")
            for key, value in model_params.items():
                print(f"  {key}: {value}")
        
        # 打印训练参数
        if training_params:
            print("\nBest Training Parameters:")
            for key, value in training_params.items():
                if key == "learning_rate":
                    print(f"  {key}: {value:.2e}")
                else:
                    print(f"  {key}: {value:.2e}")
        
        # 打印多模态融合权重（若搜索了）
        if modality_weights:
            print("\n" + "=" * 70)
            print("🎯 ACTUAL FUSION WEIGHTS USED IN TRAINING (Normalized)")
            print("=" * 70)
            
            # 归一化权重（与 objective 中的逻辑保持一致）
            total_weight = sum(modality_weights.values())
            if total_weight > 0:
                normalized_weights = {k: v / total_weight for k, v in modality_weights.items()}
            else:
                # 边界情况：所有权重为0，平均分配
                normalized_weights = {k: 1.0 / len(modality_weights) for k in modality_weights}
            
            final_total = sum(normalized_weights.values())
            # 按模态名称排序显示
            for modality in sorted(normalized_weights.keys()):
                weight = normalized_weights[modality]
                print(f"  {modality:12s}: {weight:.4f}  ({weight*100:5.2f}%)")
            print(f"  {'─' * 40}")
            print(f"  {'TOTAL':12s}: {final_total:.4f}  (100.00%)")
            print("=" * 70)

        return best_trial
