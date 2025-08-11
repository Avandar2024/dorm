# 导入必要的库
import pandas as pd
from sklearn.metrics import pairwise_distances

from fn import apply_weights, random_fill, simulated_annealing_updated
from nlp import append_ai_col


def greedy_sequence_updated(df: pd.DataFrame, dist_matrix):
    """贪婪算法获取学生序列"""
    # 以第一个学生为起点
    current_index = 0
    num_students = df.shape[0]
    sequence = [df.index[current_index]]
    used_indices = {current_index}

    while len(sequence) < num_students:
        min_distance = float('inf')
        next_index = None
        for i in range(num_students):
            # 检查未使用的学生
            if i not in used_indices and dist_matrix[current_index][i] < min_distance:
                min_distance = dist_matrix[current_index][i]
                next_index = i

        sequence.append(df.index[next_index])  # type: ignore
        used_indices.add(next_index)  # type: ignore
        current_index = next_index

    return sequence


# 读取和处理Excel文件
df = pd.read_excel('宿舍调查问卷2025_Sheet1_2025宿舍分配脱敏视图.xlsx', sheet_name=0)
df: pd.DataFrame
# 删除空白行和列
df.dropna(how='all', inplace=True)

df.drop(columns=df.columns[27:36], inplace=True)
# 暂时把后几行drop掉
# df.drop(df.index[100:], inplace=True, axis=0)
df.drop(columns=['序号'], inplace=True)

# 处理ai列
append_ai_col(df)

# 给没填的人随机赋值
random_fill(df)

# 赋权
print(f"Before apply_weights, NaNs in sub_df: {df.isna().sum().sum()}")
apply_weights(df)
print(f"After apply_weights, NaNs in sub_df: {df.isna().sum().sum()}")

# 根据性别、专业列的值划分为多个子DataFrame
grouped = df.groupby(['住宿类', '性别'])
dataframes = {}
for (category, value), group in grouped:
    if category not in dataframes:
        dataframes[category] = {}
    dataframes[category][value] = group

# 进行处理
all_dfs = []
for category, sub_dict in dataframes.items():
    for value, sub_df in sub_dict.items():
        sub_df: pd.DataFrame
        sub_df.set_index('脱敏ID', inplace=True)
        sub_df.drop(columns=['住宿类', '性别'], inplace=True)
        # 计算距离矩阵
        dist_matrix = pairwise_distances(sub_df.values, metric='euclidean')

        # 使用更新的贪婪算法获取学生序列
        student_sequence_updated = greedy_sequence_updated(sub_df, dist_matrix)

        # 创建一个从学生身份码到其在DataFrame中的位置的映射
        index_to_position_map = {index: position for position, index in enumerate(sub_df.index)}

        # 使用模拟退火算法优化学生序列
        optimized_sequence = simulated_annealing_updated(
            dist_matrix,
            student_sequence_updated,
            index_to_position_map,
        )

        # print(optimized_sequence[:10])  # 显示优化后的序列中的前10个学生
        result = {'住宿类': category, '性别': value, '脱敏ID': optimized_sequence}
        all_dfs.append(pd.DataFrame(result))

final_df = pd.concat(all_dfs)
final_df.to_excel('final_results.xlsx', index=True, engine='openpyxl')
