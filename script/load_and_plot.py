# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler

def integer_encode(seq):
    """
    Convert amino acid sequence to numerical representation.

    Args:
        seq (str): Amino acid sequence.

    Returns:
        np.ndarray: Numerical representation of the sequence.
    """
    seq = seq.upper()
    encoding = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
        'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
        'X': 21, '_': 0
    }
    return np.array([encoding.get(x, 0) for x in seq])  # Use .get(x, 0) to handle unknown characters

def seq2num(data1, data2, maxseqlen):
    """
    Convert sequences in data1 and data2 to numerical representation and save to CSV.

    Args:
        data1 (pd.DataFrame): First dataset containing sequences and labels.
        data2 (pd.DataFrame): Second dataset containing sequences and labels.
        maxseqlen (int): Maximum sequence length to pad or truncate sequences.
    """
    data = pd.concat([data1, data2], ignore_index=True)
    data['seq'] = data['seq'].apply(lambda x: str(x).ljust(maxseqlen, '_'))  # Pad sequences
    X = data['seq']
    y = data['label']
    X_encoded = X.apply(lambda x: integer_encode(x))  # Encode sequences
    X_encoded_df = X_encoded.apply(pd.Series)
    result = pd.concat([y, X_encoded_df], axis=1)
    result.to_csv('seq2num.csv', index=False, header=False)

def data_load(inputseq):
    """
    Load data from CSV, perform train-test split, and apply random oversampling.

    Args:
        inputseq (str): Path to the CSV file containing the data.

    Returns:
        tuple: (X_train_balanced, y_train_balanced, X_test, y_test)
    """
    data = pd.read_csv(inputseq)
    data_columns = data.columns.tolist()
    data = data.sample(frac=1, random_state=42).values  # Shuffle data
    row = data.shape[0]
    num_train = int(row * 0.8)  # Split data into training and testing sets
    x = data[:, 1:]
    y = data[:, 0]
    X_train = x[:num_train]
    y_train = y[:num_train]
    X_test = x[num_train:]
    y_test = y[num_train:]

    ros = RandomOverSampler(random_state=42)  # Apply random oversampling
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    X_train_balanced = pd.DataFrame(X_train_resampled, columns=data_columns[1:])
    y_train_balanced = pd.DataFrame(y_train_resampled, columns=[data_columns[0]])
    X_train_balanced.to_csv('X_train_resampled.csv', index=False)
    y_train_balanced.to_csv('y_train_resampled.csv', index=False)

    return X_train_balanced.values, y_train_balanced.values.ravel(), X_test, y_test

def read_fasta(file_path, max_len=100):
    """
    Read sequences from a FASTA file, encode them, and pad/truncate to max_len.

    Args:
        file_path (str): Path to the FASTA file.
        max_len (int): Maximum length of the sequences.

    Returns:
        list: List of encoded sequences.
    """
    sequences = []

    with open(file_path, 'r') as file:
        sequence = ''
        for line in file:
            if line.startswith('>'):  # Skip header lines
                if sequence:
                    encoded_seq = integer_encode(sequence[:max_len])
                    if len(encoded_seq) < max_len:
                        encoded_seq = np.pad(encoded_seq, (0, max_len - len(encoded_seq)), 'constant', constant_values=0)
                    sequences.append(encoded_seq)
                sequence = ''
            else:
                sequence += line.strip()
        if sequence:  # Add the last sequence
            encoded_seq = integer_encode(sequence[:max_len])
            if len(encoded_seq) < max_len:
                encoded_seq = np.pad(encoded_seq, (0, max_len - len(encoded_seq)), 'constant', constant_values=0)
            sequences.append(encoded_seq)

    return sequences

def plot_confusion(labels, predicted, path):
    """
    Plot confusion matrix and save to file.

    Args:
        labels (np.ndarray): True labels.
        predicted (np.ndarray): Predicted labels.
        path (str): Path to save the plot.
    """
    plt.rcParams.update({'font.size': 9})
    cm = confusion_matrix(labels, predicted)
    class_names = ['0', '1']
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path)

def draw_scatter(ds, names, path):
    """
    Draw scatter plot for datasets and save to file.

    Args:
        ds (list of np.ndarray): List of datasets to plot.
        names (list of str): Names of the datasets.
        path (str): Path to save the plot.
    """
    plt.rcParams.update({'font.size': 9})
    markers = ["x", "o"]
    fig, ax = plt.subplots()
    x = range(len(ds[0]))
    for d, name, marker in zip(ds, names, markers):
        ax.scatter(x, d, alpha=0.6, label=name, marker=marker)
    ax.legend(fontsize=16, loc='upper left')
    plt.savefig(path)

def plot_loss(loss_list, path):
    """
    Plot loss curve and save to file.

    Args:
        loss_list (list): List of loss values.
        path (str): Path to save the plot.
    """
    plt.rcParams.update({'font.size': 9})
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig(path)

def plot_auc_curve(labels, probabilities, path):
    """
    Plot ROC curve and save to file.

    Args:
        labels (torch.Tensor): True labels (PyTorch Tensor).
        probabilities (torch.Tensor): Predicted probabilities (PyTorch Tensor).
        path (str): Path to save the plot.
    """
    plt.rcParams.update({'font.size': 9})
    labels_np = labels.detach().numpy() if isinstance(labels, torch.Tensor) else labels
    probabilities_np = probabilities.detach().numpy() if isinstance(probabilities, torch.Tensor) else probabilities
    fpr, tpr, _ = roc_curve(labels_np, probabilities_np)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(path)