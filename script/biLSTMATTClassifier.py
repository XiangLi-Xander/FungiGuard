# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import torch.optim as optim

# Define the model
class biLSTMATTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(biLSTMATTClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention_layer = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        output, (h_n, _) = self.lstm(x)
        hn = torch.cat((h_n[0], h_n[1]), dim=1)
        hn = F.gelu(hn)
        hn = self.dropout(hn)
        attn_weights = F.softmax(self.attention_layer(hn), dim=0).unsqueeze(2)
        output = output.permute(1, 0, 2)  # Adjust dimensions for weighted sum
        attn_output = torch.sum(attn_weights * output, dim=1)
        output = self.fc(attn_output)
        return output

# Train the model
def train_model(model, X_train, y_train, num_epochs, batch_size):
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
    torch.save(model.state_dict(), '../models/bilstmatt.pth')
    return loss_list

# Evaluate the model
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X.astype(np.float64)).float().unsqueeze(1)
        labels = torch.from_numpy(y).long()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        probability = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
    return labels, predicted, probability

# Load data
data1 = pd.read_excel('../data/no.xlsx')
data2 = pd.read_excel('../data/antifu.xlsx')
maxseqlen = 100  # Maximum protein sequence length
seq2num(data1, data2, maxseqlen)
inputseq = 'seq2num.csv'
X_train, y_train, X_test, y_test = data_load(inputseq)

# Set parameters
input_size = X_train.shape[1]  # Number of features
output_size = 2  # Number of classes
hidden_size = 16
num_epochs = 1000
batch_size = 32

model = biLSTMATTClassifier(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer

loss_list = train_model(model, X_train, y_train, num_epochs, batch_size)  # Train
labels, predicted, probability = evaluate_model(model, X_test, y_test)  # Evaluate

evaluate_results = pd.DataFrame({
    'labels': labels.numpy(), 
    'predicted': predicted.numpy(), 
    'probability': probability.detach().numpy()
})
evaluate_results.to_csv('../data/biLSTMATT_evaluation_results.csv', index=False)