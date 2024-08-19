# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from load_and_plot import *  # Assumes custom functions are defined here


class LSTMClassifier(nn.Module):
    """Define a simple LSTM model for classification."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Perform a forward pass through the LSTM network."""
        output, (h_n, _) = self.lstm(x)
        hn = h_n[-1]  # Take the last hidden state
        hn = F.gelu(hn)  # Apply GELU activation function
        hn = self.dropout(hn)  # Apply dropout for regularization
        output = self.fc(hn)
        return output


def train_model(model, X_train, y_train, num_epochs, batch_size):
    """Train the LSTM model with the given training data."""
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
    
    torch.save(model.state_dict(), '../models/lstm.pth')
    return loss_list

def evaluate_model(model, X, y):
    """Evaluate the model on the test data."""
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X.astype(np.float64)).float().unsqueeze(1)
        labels = torch.from_numpy(y).long()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        probability = F.softmax(outputs, dim=1)[:, 1]  # Get probabilities for the positive class
    
    return labels, predicted, probability

# Load and preprocess data
data1 = pd.read_excel('../data/no.xlsx')
data2 = pd.read_excel('../data/antifu.xlsx')
maxseqlen = 100  # Maximum length of protein sequences
seq2num(data1, data2, maxseqlen)
inputseq = 'seq2num.csv'
X_train, y_train, X_test, y_test = data_load(inputseq)

# Model parameters
input_size = 100  # Number of input features
output_size = 2  # Number of output classes
hidden_size = 16
num_epochs = 1000
batch_size = 32

# Initialize and train the model
model = LSTMClassifier(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer
loss_list = train_model(model, X_train, y_train, num_epochs, batch_size)

# Evaluate the model
labels, predicted, probability = evaluate_model(model, X_test, y_test)

# Save evaluation results
evaluate_results = pd.DataFrame({
    'labels': labels.numpy(), 
    'predicted': predicted.numpy(), 
    'probability': probability.detach().numpy()
})
evaluate_results.to_csv('../data/LSTM_evaluation_results.csv', index=False)

# Optionally predict new data
# new_data = pd.read_excel('data/all_peps.xlsx')
# X_new = new_data_load(new_data, maxseqlen)
# predicted_labels, predicted_prob = predict_new_data(model, X_new)
# new_data['predicted_label'] = predicted_labels.numpy()
# new_data['predicted_prob'] = predicted_prob.detach().numpy()
# new_data.to_excel('data/LSTM_peps_with_predictions.xlsx', index=False)

# Optional plotting
# plot_loss(loss_list, "fig/LSTM_loss.png")
# plot_confusion(labels, predicted, "fig/LSTM_confusion.png")
# plot_auc_curve(labels, probability, "fig/LSTM_AUC.png")
# drawScatter([labels, predicted], ['true', 'pred'], "fig/LSTM_pre.png")
