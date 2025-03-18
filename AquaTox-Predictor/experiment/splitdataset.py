import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
data = pd.read_csv('../data/aquatic_toxicity_group.csv')

# 将数据分为训练集、验证集和测试集
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)  # 70% 训练集，30% 临时集
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 从临时集中 15% 验证集，15% 测试集

# 为每个数据集添加分组信息
train_data['group'] = 'training'
val_data['group'] = 'valid'
test_data['group'] = 'test'

# 合并所有数据集
grouped_data = pd.concat([train_data, val_data, test_data], ignore_index=True)

# 保存分组信息到新的 group 文件
group_file_path = '../data/aquatic_toxicity_group.csv'
grouped_data.to_csv(group_file_path, index=False)

print(f"数据集分割信息已保存到 {group_file_path}")
