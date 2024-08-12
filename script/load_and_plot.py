# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split



# 创建一个字典来映射氨基酸到数字的转换
def integer_encode(seq):
    seq = seq.upper()
    encoding = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X': 21, '_':  0
}
    return np.array([encoding[x] for x in seq])

# 转换
def seq2num(data1,data2,maxseqlen):
    data = pd.concat([data1,data2], ignore_index=True)
    data['seq'] = data['seq'].apply(lambda x: str(x).ljust(maxseqlen, '_')) #补足序列长度
    X = data['seq']
    y = data['label']
    X_encoded = X.apply(lambda x: integer_encode(x))  # 将序列进行编码
    X_encoded_df = X_encoded.apply(pd.Series)
    result = pd.concat([y, X_encoded_df], axis=1)
    result.to_csv('seq2num.csv', index=False, header=False)

def data_load(inputseq):
    data = pd.read_csv(inputseq)
    data_columns = data.columns.tolist()  # 保存列名
    data = data.sample(frac=1, random_state=42)  # frac=1 表示打乱全部数据
    data = data.values #转化为numpy
    row = data.shape[0]
    num_train = int(row * 0.8)  # 训练集与测试集划分
    x = data[:, 1:]  # 从第二列开始到最后一列的数据作为特征
    y = data[:, 0]   # 第一列数据作为标签
    X_train = x[:num_train]  # 训练集特征
    y_train = y[:num_train]  # 训练集标签
    X_test = x[num_train:]  # 测试集特征
    y_test = y[num_train:]  # 测试集标签

    ros = RandomOverSampler(random_state=42)  # 创建RandomOverSampler对象
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train) # 进行过采样
    X_train_balanced = pd.DataFrame(X_train_resampled, columns=data_columns[1:])
    y_train_balanced = pd.DataFrame(y_train_resampled, columns=[data_columns[0]])
    # 保存重采样后的结果到文件
    X_train_balanced.to_csv('X_train_resampled.csv', index=False)
    y_train_balanced.to_csv('y_train_resampled.csv', index=False)
    
    return X_train_balanced.values, y_train_balanced.values.ravel(), X_test, y_test


# 预测数据加载数据
def new_data_load(new_data, maxseqlen):
    new_data['seq'] = new_data['seq'].apply(lambda x: str(x).ljust(maxseqlen, '_')) #补足序列长度
    new_X = new_data['seq']
    new_X_encoded = new_X.apply(lambda x: integer_encode(x))  # 将序列进行编码
    X_new = new_X_encoded.apply(pd.Series)
    return X_new


# 绘制混淆热图
def plot_confusion(labels,predicted,path):
    plt.rcParams.update({'font.size': 9}) #字体
    cm = confusion_matrix(labels, predicted) # 真实标签和预测标签计算混淆矩阵
    class_names = ['0', '1']  # 设置类别标签
    fig, ax = plt.subplots() # 创建图表
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Prediction tag')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix of classification prediction results')
    plt.tight_layout() # 自动调整布局
    plt.savefig(path)
    # plt.show()

# 绘制散点图
def drawScatter(ds,names,path):
    plt.rcParams.update({'font.size': 9}) #字体
    markers = ["x", "o"]
    fig, ax = plt.subplots()
    x = range(len(ds[0]))
    for d,name,marker in zip(ds,names,markers):
        ax.scatter(x,d,alpha=0.6,label=name,marker=marker)
        ax.legend(fontsize=16, loc='upper left')
        #ax.grid(c='gray')
    plt.savefig(path)
    # plt.show()

# 绘制损失曲线
def plot_loss(loss_list,path):
    plt.rcParams.update({'font.size': 9}) #字体
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.savefig(path)
    # plt.show()

# 绘制ROC曲线
def plot_auc_curve(labels, probabilities, path):
    plt.rcParams.update({'font.size': 9}) #字体
    # 确保返回的是Tensor，将Tensor转换为NumPy数组
    labels_np = labels.detach().numpy() # 假设返回的是PyTorch Tensor
    probabilities_np = probabilities.detach().numpy()
    # 计算所有可能阈值的真阳性率和假阳性率
    fpr, tpr, thresholds = roc_curve(labels_np, probabilities_np)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
    lw=2, label='ROC Curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(path)