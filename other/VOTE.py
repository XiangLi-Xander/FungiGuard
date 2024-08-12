import pandas as pd
import glob

# 获取当前目录下所有xlsx文件的文件名
file_names = glob.glob('data/*peps_with_predictions.xlsx')

# 读取所有xlsx文件并合并成一个DataFrame
dfs = []
for file_name in file_names:
    print(file_name)
    df = pd.read_excel(file_name)
    dfs.append(df)
merged_df = pd.concat(dfs, ignore_index=True)

# 根据ID进行分组
groups = merged_df.groupby('id')

# 遍历每个分组进行投票预测
final_predictions = []
for name, group in groups:
    # 投票统计
    votes = group['predicted_label'].value_counts().to_dict()
    # 保留原始的seq、soc和len列
    seq = group['seq'].iloc[0]
    soc = group['soc'].iloc[0]
    length = group['len'].iloc[0]
    # 将得票数统计结果加入到字典中
    final_predictions.append({'ID': name, 'seq': seq, 'soc': soc, 'len': length, 'votes': votes})

# 将结果转换为DataFrame并保存到新文件中
final_df = pd.DataFrame(final_predictions)
final_df.to_csv('final_predictions.csv', index=False)