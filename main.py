# 导入必要的库
from typing import cast

from utils.decrators import static_var, WithStaticVar
from algrithorm import simulated_annealing_updated
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import locally_linear_embedding

"""住宿类
    性别
    早上七点及以后
    早上六点到七点
    早上六点及以前
    2. 如果第一节课程时间在九点以前，您能接受的舍友前一天晚上睡眠时间为：
    3. 不考虑课程因素，您能接受的舍友起床时间为：
    4. 不考虑课程因素，您能接受的舍友睡觉时间为：
    5. 在工作日，您是否有午休的习惯？
    6. 睡觉的时候，您对周围环境的灯光是否敏感？
    7. 睡觉的时候，您对周围环境的声音是否敏感？
    8. 睡觉的时候，您能最多能容忍室友发出多大的噪音：
    9. 您认为夏天空调应该开多少度？
    10. 您认为冬天宿舍空调应该开多少度：（宿舍没有暖气）
    11. 如果当天开了空调，您认为开多久合适：
    12. 夏天，您能忍受舍友多久不洗澡？
    13. 冬天，您能忍受舍友多久不洗澡？
    14. 您能忍受舍友多久不洗衣服/袜子：
    15. 您是否能接受舍友在宿舍吃正餐：（外卖、速食、食堂打包的饭菜等）
    16. 您是否能接受宿舍有烟味：
    17. 在学习或工作时，您是否能接受舍友在室内大声喧哗：（如开麦，打电话，看视频等）
    18. 您是否能接受舍友在室内读书、唱歌、听课时外放声音：
    19. 您可以接受并做到的打扫卫生频率：
    20. 假设今天您和舍友一起出去吃饭、娱乐，在涉及到消费的问题上，您会更希望：
    21. 当您的朋友喊你去参加集体活动，比如一起旅行或者聚会，您的态度是：
    22. 您是否能接受舍友带朋友来宿舍做客："""


def apply_weights(df):
    # 根据22级数据进行赋权
    """作息: 0.441099
    卫生习惯: 0.200818
    噪音: 0.166033
    烟味: 0.129494
    空调: 0.032154
    消费: 0.030400"""
    weights = [0.441099, 0.200818, 0.166033, 0.129494, 0.032154, 0.030400]
    for index, col_name in enumerate(df.columns):
        df[col_name] *= 100
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
        else:
            df[col_name] *= weights[5]
    return df


def encoding(df):
    # 对其它列使用Label-Encoder进行简单编码
    label_encoders = {}
    for col in df.columns[1:]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    df = df.astype(float)
    return df


def confidence_level():
    """计算置信度"""
    pass


@static_var(value=None)
def dist_matrix(df: pd.DataFrame) -> np.ndarray:
    """计算距离矩阵，使用静态缓存"""

    dist_matrix_proxy = cast(WithStaticVar, dist_matrix)

    if dist_matrix_proxy.value is None:
        print("首次计算距离矩阵...")
        dist_matrix_proxy.value = pairwise_distances(df.to_numpy(), metric="euclidean")

    return dist_matrix_proxy.value


def greedy_sequence_updated(df: pd.DataFrame):
    """贪婪算法获取学生序列"""
    # 以第一个学生为起点
    current_index = 0
    num_students = df.shape[0]
    sequence = [df.index[current_index]]
    used_indices = {current_index}

    while len(sequence) < num_students:
        min_distance = float("inf")
        next_index = None
        for i in range(num_students):
            # 检查未使用的学生
            if i not in used_indices:
                if dist_matrix(df)[current_index][i] < min_distance:
                    min_distance = dist_matrix(df)[current_index][i]
                    next_index = i

        sequence.append(df.index[next_index])  # type: ignore
        used_indices.add(next_index)  # type: ignore
        current_index = next_index

    return sequence


if __name__ == "__main__":
    # 读取和处理Excel文件
    df = pd.read_excel("0814导出-加密_Sheet1_1.xlsx", sheet_name=0)
    # 对第一列进行多选编码
    col1 = df.columns[3]
    options = ["早上六点及以前", "早上六点到七点", "早上七点及以后"]
    for option in options:
        df.insert(4, option, df[col1].str.contains(option).astype(int))
    df.drop(columns=col1, inplace=True)
    # print(df.head())

    # 根据性别、专业列的值划分为多个子DataFrame
    grouped = df.groupby(["住宿类", "性别"])
    dataframes = {}
    for (category, value), group in grouped:
        if category not in dataframes:
            dataframes[category] = {}
        dataframes[category][value] = group

    # 进行处理
    all_dfs = []
    for category, sub_dict in dataframes.items():
        for value, sub_df in sub_dict.items():
            sub_df.set_index("身份码", inplace=True)
            sub_df.drop(columns=["住宿类", "性别"], inplace=True)

            # 进行编码和赋权
            encoding(sub_df)
            apply_weights(sub_df)

            # 使用更新的贪婪算法获取学生序列
            student_sequence_updated = greedy_sequence_updated(sub_df)

            # 创建一个从学生身份码到其在DataFrame中的位置的映射
            index_to_position_map = {
                index: position for position, index in enumerate(sub_df.index)
            }

            # 使用模拟退火算法优化学生序列
            optimized_sequence = simulated_annealing_updated(
                dist_matrix(sub_df),
                student_sequence_updated,
                index_to_position_map,
            )

            # print(optimized_sequence[:10])  # 显示优化后的序列中的前10个学生
            result = {"住宿类": category, "性别": value, "身份码": optimized_sequence}
            all_dfs.append(pd.DataFrame(result))

    final_df = pd.concat(all_dfs)
    final_df.to_excel("final_results.xlsx", index=True, engine="openpyxl")
