"""多染色融合策略的共享工具函数，供 multi_stain_fusion.py 调用。

此模块已从核心库移出，作为实验性功能维护。
"""

# 标准库
import os
import sys
from typing import Any, Dict, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 第三方库
import numpy as np
import torch

# 项目内部模块
from PathoML.config.config import runtime_config
from PathoML.optimization.registry import create_dataset, create_model, load_runtime_plugins
from PathoML.optimization.trainer import Trainer

# ANSI color helpers for terminal clarity
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def build_dataset():
    """根据 runtime_config 构建数据集并自动设置分类数。

    读取全局 ``runtime_config.dataset`` 配置，通过工厂函数创建数据集实例，
    并将 ``runtime_config.model.num_classes`` 自动设置为 1（二分类）或实际类别数。

    Returns:
        构建好的数据集对象（BaseDataset 子类实例）。
    """
    load_runtime_plugins(runtime_config)
    dataset_cfg = runtime_config.dataset
    dataset_kwargs = dict(dataset_cfg.dataset_kwargs)
    dataset_kwargs.setdefault("patient_id_pattern", dataset_cfg.patient_id_pattern)
    dataset_kwargs.setdefault("binary_mode", dataset_cfg.binary_mode)
    dataset = create_dataset(dataset_cfg.dataset_name, **dataset_kwargs)
    num_dataset_classes = len(dataset.classes)
    runtime_config.model.num_classes = 1 if num_dataset_classes == 2 else num_dataset_classes
    return dataset


def build_model_builder():
    """根据 runtime_config 创建模型构造工厂函数。

    返回一个无参可调用对象（闭包），每次调用时根据当前
    ``runtime_config.model`` 配置实例化并返回一个新模型。
    这种工厂模式便于在交叉验证的每个 fold 中重新初始化模型权重。

    Returns:
        可调用对象，每次调用返回一个新初始化的模型实例。
    """
    load_runtime_plugins(runtime_config)
    model_cfg = runtime_config.model

    def _builder():
        return create_model(
            model_cfg.model_name,
            input_dim=model_cfg.input_dim,
            hidden_dim=model_cfg.hidden_dim,
            num_classes=model_cfg.num_classes,
            dropout=model_cfg.dropout,
            attention_dim=model_cfg.attention_dim,
            gated=model_cfg.gated,
            encoder_dropout=model_cfg.encoder_dropout,
            classifier_dropout=model_cfg.classifier_dropout,
        )

    return _builder


def run_cv_for_stain(stain_cfg: Dict[str, Any], k_folds: int, save_features: bool = False):
    """对单个染色运行完整的交叉验证流程。

    Args:
        stain_cfg: 染色配置字典，包含 ``name``、``data_paths``、``save_dir``、``weight`` 字段。
        k_folds: 交叉验证折数。
        save_features: 若为 ``True``，则在每折结束后保存特征供融合使用。

    Returns:
        ``trainer.cross_validate()`` 的返回值（各折指标列表）。
    """
    runtime_config.dataset.dataset_kwargs["data_paths"] = stain_cfg["data_paths"]
    runtime_config.logging.save_dir = stain_cfg["save_dir"]
    runtime_config.logging.save_features = save_features

    print(f"\n{CYAN}================ Running stain: {stain_cfg['name']} ================{RESET}")
    dataset = build_dataset()
    if len(dataset) == 0:
        raise RuntimeError(f"No data found for stain {stain_cfg['name']}")

    model_builder = build_model_builder()
    trainer = Trainer(model_builder, dataset, runtime_config)
    return trainer.cross_validate(k_folds=k_folds)


def get_common_patients(stain_runs: List[Dict], k_folds: int, split: str = 'test') -> Dict[int, set]:
    """获取各折中所有染色共同拥有的患者 ID 集合。

    Args:
        stain_runs: 染色配置字典列表。
        k_folds: 交叉验证折数。
        split: 数据集划分类型，``'train'`` 或 ``'test'``。

    Returns:
        字典，键为折编号（从 1 起），值为该折下所有染色共有的患者 ID 集合。
    """
    common_patients = {}
    
    for fold in range(1, k_folds + 1):
        fold_patients = None
        
        for run in stain_runs:
            features_path = os.path.join(run["save_dir"], f'fold_{fold}_features.pt')
            if not os.path.exists(features_path):
                print(f"{YELLOW}Warning: Missing features for stain {run['name']} fold {fold}{RESET}")
                continue
                
            data = torch.load(features_path, weights_only=False)
            patient_ids = set(data[split]['patient_ids'])
            
            if fold_patients is None:
                fold_patients = patient_ids
            else:
                fold_patients = fold_patients.intersection(patient_ids)
        
        common_patients[fold] = fold_patients if fold_patients else set()
    
    return common_patients


def load_and_align_features(
    stain_runs: List[Dict],
    fold: int,
    split: str = 'test',
    use_all_patients: bool = True
) -> Dict[str, Any]:
    """从所有染色加载指定折的特征，并按患者 ID 对齐，缺失模态以零填充。

    Args:
        stain_runs: 染色配置字典列表。
        fold: 折编号（从 1 起）。
        split: 数据集划分类型，``'train'`` 或 ``'test'``。
        use_all_patients: 若为 ``True``，取所有患者的并集（缺失染色用零填充）；
                          若为 ``False``，取所有染色均存在的患者交集。

    Returns:
        字典，包含：

        - ``patient_ids``: 患者 ID 列表
        - ``features``: 形状为 (n_patients, total_feature_dim) 的特征矩阵
        - ``labels``: 形状为 (n_patients,) 的标签数组
        - ``stain_dims``: 各染色特征维度字典
        - ``available_stains``: 患者 ID → 可用染色列表的字典
    """
    # 第一遍扫描：收集所有数据并确定各染色的特征维度
    stain_data = {}
    stain_dims = {}
    all_patient_ids = set()
    patient_labels = {}
    
    for run in stain_runs:
        features_path = os.path.join(run["save_dir"], f'fold_{fold}_features.pt')
        if not os.path.exists(features_path):
            print(f"{YELLOW}Warning: Missing features for stain {run['name']} fold {fold}{RESET}")
            continue
            
        data = torch.load(features_path, weights_only=False)
        split_data = data[split]
        
        stain_name = run["name"]
        stain_dims[stain_name] = split_data['features'].shape[1]
        
        # 建立患者 -> 特征的映射
        stain_data[stain_name] = {}
        for i, pid in enumerate(split_data['patient_ids']):
            stain_data[stain_name][pid] = split_data['features'][i]
            all_patient_ids.add(pid)
            # 存储标签（各染色对同一患者的标签应保持一致）
            if pid not in patient_labels:
                patient_labels[pid] = split_data['labels'][i]
    
    if not stain_data:
        raise RuntimeError(f"No feature data loaded for fold {fold}")
    
    # 确定最终纳入的患者范围
    if use_all_patients:
        # 并集：纳入所有患者，缺失染色以零向量填充
        final_patient_ids = sorted(all_patient_ids)
    else:
        # 交集：仅保留所有染色均有数据的患者
        final_patient_ids = sorted(all_patient_ids)
        for stain_name, data in stain_data.items():
            final_patient_ids = [pid for pid in final_patient_ids if pid in data]

    # 总特征维度 = 各染色特征维度之和
    total_dim = sum(stain_dims.values())
    stain_order = list(stain_dims.keys())

    # 第二遍遍历：构建对齐的特征矩阵（缺失染色以零填充）
    n_patients = len(final_patient_ids)
    features = np.zeros((n_patients, total_dim), dtype=np.float32)
    labels = np.zeros(n_patients, dtype=np.int64)
    available_stains = {}

    for i, pid in enumerate(final_patient_ids):
        available_stains[pid] = []
        offset = 0

        for stain_name in stain_order:
            dim = stain_dims[stain_name]

            if stain_name in stain_data and pid in stain_data[stain_name]:
                features[i, offset:offset+dim] = stain_data[stain_name][pid]
                available_stains[pid].append(stain_name)
            # else: 保持零向量（缺失模态填充）

            offset += dim

        labels[i] = patient_labels.get(pid, 0)
    
    return {
        'patient_ids': final_patient_ids,
        'features': features,
        'labels': labels,
        'stain_dims': stain_dims,
        'stain_order': stain_order,
        'available_stains': available_stains
    }
