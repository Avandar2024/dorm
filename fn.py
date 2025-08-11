import math
import random
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd


def num_in_title(title: str) -> int:
    """从标题中提取数字"""
    num = ''.join(filter(str.isdigit, title))
    return int(num) if num else 0


@dataclass
class col_wrapper:
    max_value: int = 0
    col: str = ''
    src_df: pd.DataFrame | None = None

    def random_fill(self):
        """随机填充DataFrame中的空值"""
        if self.src_df is None or self.col not in self.src_df.columns:
            raise ValueError('DataFrame or column is not properly initialized.')
        self.src_df[self.col] = self.src_df[self.col].fillna(
            np.random.randint(0, self.max_value + 1),
        )


@cache
def ai_weight() -> float:
    """计算ai的置信度"""
    max_level = 0.3
    min_level = 0.2
    level = 0.2
    return level


def apply_weights(df: pd.DataFrame):
    # 根据22级数据进行赋权
    """
    作息: 0.441099
    卫生习惯: 0.200818
    噪音: 0.166033
    烟味: 0.129494
    空调: 0.032154
    """
    weights = [0.441099, 0.200818, 0.166033, 0.129494, 0.032154]
    weights = [i * (1 - ai_weight()) for i in weights]
    weights.append(ai_weight())
    for col_name in df.columns:
        col_name: str
        if not pd.api.types.is_numeric_dtype(df[col_name]):
            print(f'Skipping non-numeric column: {col_name}')
            print(f'Column content: {df[col_name].head()}')
            continue
        # 归一化
        df[col_name] = (df[col_name] - 1) / (df[col_name].max() - 1)
        df[col_name] = (df[col_name] * 100).astype(float)
        index = num_in_title(col_name)
        if index <= 8:
            df[col_name] *= weights[0]
        elif index <= 11 or index == 20 or index == 21:
            df[col_name] *= weights[2]
        elif index <= 14:
            df[col_name] *= weights[4]
        elif index <= 18 or index == 22:
            df[col_name] *= weights[1]
        elif index == 19:
            df[col_name] *= weights[3]
        elif index == 26:
            df[col_name] *= weights[5]
    return df


def cost(sequence, dist_matrix, index_to_position_map):
    """计算序列的代价函数：任意相邻四个学生的距离之和"""
    cost_value = 0
    for i in range(len(sequence) - 3):
        for j in range(i, i + 4):
            for k in range(j + 1, i + 4):
                idx1 = index_to_position_map[sequence[j]]
                idx2 = index_to_position_map[sequence[k]]
                cost_value += dist_matrix[idx1][idx2]
    return cost_value


def simulated_annealing_updated(
    dist_matrix,
    init_sequence,
    index_to_position_map,
    init_temp=100,
    end_temp=0.1,
    alpha=0.995,
    max_iters=10000,
):
    """模拟退火算法优化"""
    current_sequence = init_sequence.copy()
    current_cost = cost(current_sequence, dist_matrix, index_to_position_map)
    best_sequence = current_sequence
    best_cost = current_cost

    temp = init_temp

    for _ in range(max_iters):
        # 生成新的解决方案：交换两个随机位置的学生
        new_sequence = current_sequence.copy()
        idx1, idx2 = random.sample(range(len(new_sequence)), 2)
        new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]

        new_cost = cost(new_sequence, dist_matrix, index_to_position_map)

        # 如果新的代价函数值更低，或者满足一定的概率准则，接受新的序列
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temp):
            current_sequence = new_sequence
            current_cost = new_cost
            # 更新最佳序列
            if current_cost < best_cost:
                best_cost = current_cost
                best_sequence = current_sequence

        # 降低温度
        temp *= alpha
        if temp < end_temp:
            break

    return best_sequence


def random_fill(df: pd.DataFrame):
    """随机填充DataFrame中的空值"""
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        max_value = df[col].max(skipna=True)
        if pd.isna(max_value):
            print(f'Column {col} has no valid max value, skipping.')
            print(f'Column content: {df[col].head()}')
        col_wrapper(max_value, col, df).random_fill()
