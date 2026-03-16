"""患者级别投票策略：将同一患者的多张切片预测聚合为最终患者级别预测。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def register_voting(_key: str):
    """遗留脚本内部装饰器：保持历史代码可运行，不写入核心注册表。"""

    def _decorator(target):
        return target

    return _decorator


class BaseVotingStrategy(ABC):
    """患者级别预测聚合策略的抽象基类。

    子类需实现 ``aggregate`` 方法，将切片级别（slide-level）的预测结果
    聚合为患者级别（patient-level）的最终预测。
    """

    def __init__(self, **kwargs) -> None:
        """初始化投票策略，保存额外配置参数。

        Args:
            **kwargs: 策略相关的配置参数，例如 threshold（分类阈值）。
        """
        self.params = kwargs

    @abstractmethod
    def aggregate(self, results_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
        raise NotImplementedError


@register_voting('average')
class AverageVoting(BaseVotingStrategy):
    """对患者所有切片的预测概率取平均后进行分类。"""

    def aggregate(self, results_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
        """将切片级别概率按患者取平均，并以阈值进行二值化预测。

        Args:
            results_df: 切片级别预测结果，需包含 'patient_id'、'prob_positive'
                        （二分类）或 'prob_class_*'（多分类）列。
            num_classes: 分类数。为 1 时执行二分类逻辑，否则执行多分类逻辑。

        Returns:
            患者级别聚合结果 DataFrame，包含 'patient_id'、'prediction'
            以及各类别的平均概率列。
        """
        if num_classes == 1:
            grouped = results_df.groupby('patient_id')['prob_positive'].mean().reset_index()
            threshold = self.params.get('threshold', 0.5)
            grouped['prediction'] = (grouped['prob_positive'] > threshold).astype(int)
            return grouped

        prob_cols = [c for c in results_df.columns if c.startswith('prob_class_')]
        grouped = results_df.groupby('patient_id')[prob_cols].mean().reset_index()
        grouped['prediction'] = grouped[prob_cols].values.argmax(axis=1)
        return grouped


@register_voting('majority')
class MajorityVoting(BaseVotingStrategy):
    """基于多数投票的患者级别聚合策略，以平均概率作为平局时的决胜依据。"""

    def aggregate(self, results_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
        """对每位患者的所有切片进行多数投票，得票相同时以平均概率决胜。

        Args:
            results_df: 切片级别预测结果，需包含 'patient_id'、'prediction'、
                        'prob_positive'（二分类）或 'prob_class_*'（多分类）列。
            num_classes: 分类数。为 1 时执行二分类逻辑，否则执行多分类逻辑。

        Returns:
            患者级别聚合结果 DataFrame，包含 'patient_id'、'prediction'
            以及平均概率列。
        """
        threshold = self.params.get('threshold', 0.5)
        records = []
        for patient_id, group in results_df.groupby('patient_id'):
            if num_classes == 1:
                vote_sum = group['prediction'].sum()
                total = len(group)
                if vote_sum > total / 2:
                    prediction = 1
                elif vote_sum < total / 2:
                    prediction = 0
                else:
                    prediction = int(group['prob_positive'].mean() > threshold)
                records.append(
                    {
                        'patient_id': patient_id,
                        'prediction': prediction,
                        'prob_positive': group['prob_positive'].mean(),
                    }
                )
                continue

            counts = group['prediction'].value_counts()
            top = counts[counts == counts.max()].index.tolist()
            if len(top) == 1:
                prediction = int(top[0])
            else:
                prob_cols = [c for c in group.columns if c.startswith('prob_class_')]
                summed = group[prob_cols].sum()
                prediction = int(summed.idxmax().replace('prob_class_', ''))

            record = {'patient_id': patient_id, 'prediction': prediction}
            prob_cols = [c for c in group.columns if c.startswith('prob_class_')]
            if prob_cols:
                record.update(group[prob_cols].mean().to_dict())
            records.append(record)

        return pd.DataFrame(records)


@register_voting('max_confidence')
class MaxConfidenceVoting(BaseVotingStrategy):
    """最高置信度优先策略：任意切片对某类的置信度超过阈值即判为该类阳性。

    临床直觉：同一患者多个组织块中，只要任意一个显示出高置信度的阳性信号，
    即应将该患者判为阳性——而非因其他低信号切片"平均"掉该信号。
    这与病理诊断实践一致：一处发现即足以确诊。
    """

    def aggregate(self, results_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
        """取各切片中最高的阳性概率作为患者级别概率，超过阈值则判阳性。

        对二分类：``patient_prob = max(slide_prob_positive)``，超过 threshold 则预测为 1。
        对多分类：每个类别取所有切片中的最高概率，再对各类别做 argmax。

        Args:
            results_df: 切片级别预测结果，需包含 'patient_id'、'prob_positive'
                        （二分类）或 'prob_class_*'（多分类）列。
            num_classes: 分类数。为 1 时执行二分类逻辑，否则执行多分类逻辑。

        Returns:
            患者级别聚合结果 DataFrame，包含 'patient_id'、'prediction'
            以及最高概率列。
        """
        threshold = self.params.get('threshold', 0.5)
        records = []

        for patient_id, group in results_df.groupby('patient_id'):
            if num_classes == 1:
                # 取所有切片中最高的阳性概率：任一切片高置信阳性则患者为阳性
                max_prob = group['prob_positive'].max()
                prediction = int(max_prob > threshold)
                records.append({
                    'patient_id': patient_id,
                    'prediction': prediction,
                    'prob_positive': max_prob,
                })
            else:
                # 多分类：对每个类别取所有切片中的最高概率，再 argmax
                prob_cols = [c for c in group.columns if c.startswith('prob_class_')]
                max_probs = group[prob_cols].max()   # 每个类别的最高概率
                prediction = int(max_probs.values.argmax())
                record = {'patient_id': patient_id, 'prediction': prediction}
                record.update(max_probs.to_dict())
                records.append(record)

        return pd.DataFrame(records)
