import pandas as pd

# 创建一个示例 DataFrame
data = {'A': ['apple', 'orange', 'apple', 'banana'],
        'B': [1, 2, 3, 4]}
df = pd.DataFrame(data)
col = df.iloc[:, -1]  # 获取第一列
df.drop(columns=df.columns[-1], inplace=True)  # 删除第一列
print(df)