# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from load_and_plot import *
import seaborn as sns
import torch.optim as optim

# 定义模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        output, (h_n, _) = self.lstm(x)
        # 将双向LSTM的隐藏状态拼接起来
        hn = torch.cat((h_n[0], h_n[1]), dim=1)
        # 应用GELU激活函数
        hn = F.gelu(hn)
        # 添加Dropout层
        hn = self.dropout(hn)
        output = self.fc(hn)
        return output


# 训练模型
def train_model(model, X_train, y_train, num_epochs,batch_size):
    loss_list = []
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            inputs = torch.from_numpy(X_train[i:i + batch_size].astype(np.float64)).float().unsqueeze(1)
            labels = torch.from_numpy(y_train[i:i + batch_size]).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        loss_list.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    torch.save(model.state_dict(),"../models/LSTMmodel.pth")

    return loss_list

# 评估
def evaluate_model(model, X, y,name):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X.astype(np.float64)).float().unsqueeze(1)
        labels = torch.from_numpy(y).long()
        outputs = model(inputs)
        # print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        probability = torch.nn.functional.softmax(outputs, dim=1)[:, 1]  # 获取预测为正例的，softmax进行统计

    # print(' {} Accuracy: {:.2f}%'.format(name,accuracy * 100))
    # print("labels:",labels)
    # print("predicted:",predicted)
    return labels,predicted,probability

# 预测
def predict_new_data(model, X_new):
    inputs = torch.from_numpy(X_new.values.astype(np.float64)).float().unsqueeze(1)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    probability = torch.nn.functional.softmax(outputs, dim=1)[:, 1]  # 获取预测为正例的，softmax进行统计
    
    return predicted, probability



# data1 = pd.read_excel('data/no_remaining.xlsx')
# data2 = pd.read_excel('data/antifu_remaining.xlsx')
data1 = pd.read_excel('../data/no.xlsx')
data2 = pd.read_excel('../data/antifu.xlsx')
maxseqlen = 100 # 最长的蛋白序列长度
seq2num(data1,data2,maxseqlen)
inputseq ='seq2num.csv'
X_train,y_train,X_test,y_test=data_load(inputseq)
# 设置参数
input_size = X_train.shape[1]  #特征数
output_size = 2    #分类数
# output_size = len(np.unique(y_train)) #分类数
hidden_size = 16
num_epochs = 1000
batch_size = 32
print("input_size,output_size",input_size,output_size)
model = LSTMClassifier(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss() #损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #优化器
loss_list = train_model(model, X_train, y_train, num_epochs,batch_size) #训练
labels,predicted,probability=evaluate_model(model, X_test, y_test,"X_test") #评估测试集
#labels,predicted=evaluate_model(model, X_train, y_train,"X_train") #评估训练集
evaluate_results = pd.DataFrame({'labels': labels.numpy(), 
    'predicted': predicted.numpy(), 
    'probability': probability.detach().numpy()})
evaluate_results.to_csv('data/biLSTM_evaluation_results.csv', index=False)

# # 使用模型进行预测 # 包括两列，分别是id_seq
# new_data = pd.read_excel('data/all_peps.xlsx')
# X_new = new_data_load(new_data, maxseqlen)
# predicted_labels, predicted_prob = predict_new_data(model, X_new)
# new_data['predicted_label'] = predicted_labels.numpy()
# new_data['predicted_prob'] = predicted_prob.detach().numpy()
# new_data.to_excel('data/biLSTM_peps_with_predictions.xlsx', index=False)

# # 绘图
# plot_loss(loss_list,"fig/biLSTM_confusion.png")
# plot_confusion(labels,predicted,"fig/biLSTM_loss.png")
# plot_auc_curve(labels, probability, "fig/biLSTM_AUC.png")
# # drawScatter([labels, predicted], ['true', 'pred'],"fig/LSTM_pre.png")