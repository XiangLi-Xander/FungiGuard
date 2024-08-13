import pickle
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
import sys
# 从模型文件导入模型类
from biLSTMATTClassifier import biLSTMATTClassifier
from biLSTMClassifier import LSTMClassifier
from LSTMATTClassifier import LSTMATTClassifier
from LSTMClassifier import LSTMClassifier

# 将蛋白质序列转换为数值表示的函数
def integer_encode(seq):
    seq = seq.upper()
    encoding = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
        'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
        'X': 21, '_': 0
    }
    return np.array([encoding.get(x, 21) for x in seq])

# 预测数据加载和处理函数
def new_data_load(new_data, maxseqlen):
    new_data['seq'] = new_data['seq'].apply(lambda x: str(x).ljust(maxseqlen, '_'))  # 补足序列长度
    new_X = new_data['seq']
    new_X_encoded = new_X.apply(lambda x: integer_encode(x))  # 将序列进行编码
    X_new = np.vstack(new_X_encoded.apply(pd.Series).values)  # 将每个编码后的序列转换为行并合并为二维数组
    return X_new

# 加载模型
def load_models(model_paths):
    models = []
    for path in model_paths:
        if 'RandomForest' in path:
            # 对于 RandomForest 模型
            with open(path, 'rb') as f:
                model = pickle.load(f)
        else:
            # 对于 PyTorch 模型
            if 'biLSTMATT' in path:
                model = BiLSTMATTClassifier()
            elif 'biLSTM' in path:
                model = BiLSTMClassifier()
            elif 'LSTMATT' in path:
                model = LSTMATTClassifier()
            elif 'LSTM' in path:
                model = LSTMClassifier()
            else:
                raise ValueError(f"Unknown model type for path: {path}")
            state_dict = torch.load(path)
            model.load_state_dict(state_dict)
        models.append(model)
    return models

# 预测新数据
def predict_with_models(models, X_new):
    results = []
    for model in models:
        model.eval()
        if isinstance(model, RandomForestClassifier):
            # 对于 Scikit-learn 模型
            outputs = model.predict(X_new)
            probabilities = model.predict_proba(X_new)[:, 1]
        else:
            # 对于 PyTorch 模型
            X_new_tensor = torch.tensor(X_new, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(X_new_tensor).numpy()
                probabilities = torch.softmax(torch.tensor(outputs), dim=1)[:, 1].numpy()
                outputs = (probabilities > 0.5).astype(int)
        results.append((outputs, probabilities))
    return results

# 读取FA文件并提取序列
def read_fa_file(fa_file):
    sequences = []
    for record in SeqIO.parse(fa_file, "fasta"):
        if len(record.seq) > 100:
            raise ValueError(f"Sequence length exceeds 100 residues: {record.id}")
        sequences.append(str(record.seq))
    return pd.DataFrame({'seq': sequences})

# 主程序
def main(fa_file):
    model_paths = [
        '../models/biLSTMATT.pkl',
        '../models/LSTMATT.pkl',
        '../models/LSTMmodel.pkl',
        '../models/biLSTMmodel.pkl',
        '../models/RandomForestModel.pkl'
    ]  # 模型文件路径
    output_excel_file = 'prediction_results.xlsx'
    maxseqlen = 100  # 最大序列长度

    # 读取FA文件
    new_data = read_fa_file(fa_file)

    # 处理新数据
    X_new = new_data_load(new_data, maxseqlen)

    # 加载模型
    models = load_models(model_paths)

    # 预测
    results = predict_with_models(models, X_new)

    # 保存结果到Excel文件
    with pd.ExcelWriter(output_excel_file) as writer:
        for i, (outputs, probabilities) in enumerate(results):
            df = pd.DataFrame({
                'Sequence': new_data['seq'],
                'Predicted Class': outputs,
                'Probability of being AFP': probabilities
            })
            df.to_excel(writer, sheet_name=f'Model_{i+1}', index=False)

    # 输出每个模型的分类结果
    for i, (outputs, _) in enumerate(results):
        print(f"Model {i+1} Classification Counts:")
        print(f"Class 1: {(outputs == 1).sum()}")
        print(f"Class 0: {(outputs == 0).sum()}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_fa_file>")
        sys.exit(1)
    
    fa_file = sys.argv[1]
    main(fa_file)
