import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from load_and_plot import *
import seaborn as sns
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend

# 训练模型
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# 评估
def evaluate_model(model, X, y, name):
    outputs = model.predict(X)
    accuracy = (outputs == y).sum().item() / len(y)
    probabilities = model.predict_proba(X)[:, 1]  # 获取预测为正例的概率值
    return torch.tensor(y), torch.tensor(outputs), torch.tensor(probabilities)

# 预测
def predict_new_data(model, X_new):
    outputs = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]
    return torch.tensor(outputs), torch.tensor(probabilities)

data1 = pd.read_excel('data/no.xlsx')
data2 = pd.read_excel('data/antifu.xlsx')
# data1 = pd.read_excel('data/no_remaining.xlsx')
# data2 = pd.read_excel('data/antifu_remaining.xlsx')
maxseqlen = 100
seq2num(data1, data2, maxseqlen)
inputseq ='seq2num.csv'
X_train, y_train, X_test, y_test = data_load(inputseq)

# 设置参数
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}

with parallel_backend('threading', n_jobs=-1):
    model = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=3)
    model = train_model(model, X_train, y_train)

labels, predicted, probabilities = evaluate_model(model, X_test, y_test, "X_test")
evaluate_results = pd.DataFrame({'labels': labels.numpy(), 
                                  'predicted': predicted, 
                                  'probability': probabilities.numpy()})

evaluate_results.to_csv('data/SVM_evaluation_results.csv', index=False)

# new_data = pd.read_excel('data/all_peps.xlsx')
# new_data = pd.read_excel('data/peps_验证.xlsx')
# X_new = new_data_load(new_data, maxseqlen)
# predicted_labels, predicted_prob = predict_new_data(model, X_new)
# new_data['predicted_label'] = predicted_labels
# new_data['predicted_prob'] = predicted_prob.numpy()
# new_data.to_excel('data/SVM_peps_with_predictions.xlsx', index=False)

# # 绘图
# # plot_loss(None, "fig/SVM_loss.png")
# plot_confusion(labels, predicted, "fig/SVM_confusion.png")
# plot_auc_curve(labels, probabilities, "fig/SVM_AUC.png")