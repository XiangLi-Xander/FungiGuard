import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from load_and_plot import *

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

# 读取数据
data1 = pd.read_excel('../data/no.xlsx')
data2 = pd.read_excel('../data/antifu.xlsx')
# data1 = pd.read_excel('data/no_remaining.xlsx')
# data2 = pd.read_excel('data/antifu_remaining.xlsx')
maxseqlen = 100
seq2num(data1, data2, maxseqlen)
inputseq = 'seq2num.csv'
X_train, y_train, X_test, y_test = data_load(inputseq)

# 设置参数
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 实例化模型
model = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=3)
model = train_model(model, X_train, y_train)

# 保存训练好的模型
with open('../models/RandomForestModel.pkl', 'wb') as f:
    pickle.dump(model, f)

# 评估模型
labels, predicted, probabilities = evaluate_model(model, X_test, y_test, "X_test")
evaluate_results = pd.DataFrame({
    'labels': labels.numpy(),
    'predicted': predicted.numpy(),
    'probability': probabilities.numpy()
})

evaluate_results.to_csv('RandomForest_evaluation_results.csv', index=False)

# 使用模型进行预测
# new_data = pd.read_excel('data/all_peps.xlsx')
# X_new = new_data_load(new_data, maxseqlen)
# predicted_labels, predicted_prob = predict_new_data(model, X_new)
# new_data['predicted_label'] = predicted_labels.numpy()
# new_data['predicted_prob'] = predicted_prob.numpy()
# new_data.to_excel('data/RandomForest_peps_with_predictions.xlsx', index=False)

# # 绘图
# # plot_loss(None, "fig/RandomForest_confusion.png")
# plot_confusion(labels, predicted, "fig/RandomForest_loss.png")
# plot_auc_curve(labels, probabilities, "fig/RandomForest_AUC.png")