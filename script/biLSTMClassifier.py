# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from load_and_plot import *
import seaborn as sns
import torch.optim as optim

class biLSTMClassifier(nn.Module):
    """Define a bidirectional LSTM classifier model."""
    def __init__(self, input_size, hidden_size, output_size):
        super(biLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """Perform forward pass through the model."""
        output, (h_n, _) = self.lstm(x)
        hn = torch.cat((h_n[0], h_n[1]), dim=1)
        hn = F.gelu(hn)
        hn = self.dropout(hn)
        output = self.fc(hn)
        return output

def train_model(model, X_train, y_train, num_epochs, batch_size):
    """Train the biLSTM model with the given training data."""
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
    
    torch.save(model.state_dict(), '../models/bilstm.pth')
    return loss_list

def evaluate_model(model, X, y):
    """Evaluate the biLSTM model on the test data."""
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X.astype(np.float64)).float().unsqueeze(1)
        labels = torch.from_numpy(y).long()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
    
    return labels, predicted, probability

# Prepare data
data1 = pd.read_excel('../data/no.xlsx')
data2 = pd.read_excel('../data/antifu.xlsx')
maxseqlen = 100
seq2num(data1, data2, maxseqlen)
inputseq = 'seq2num.csv'
X_train, y_train, X_test, y_test = data_load(inputseq)

# Set model parameters
input_size = X_train.shape[1]
output_size = 2
hidden_size = 16
num_epochs = 1000
batch_size = 32

# Initialize and train the model
model = biLSTMClassifier(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_list = train_model(model, X_train, y_train, num_epochs, batch_size)

# Evaluate the model
labels, predicted, probability = evaluate_model(model, X_test, y_test)

# Save evaluation results
evaluate_results = pd.DataFrame({
    'labels': labels.numpy(), 
    'predicted': predicted.numpy(), 
    'probability': probability.detach().numpy()
})
evaluate_results.to_csv('../data/biLSTM_evaluation_results.csv', index=False)
