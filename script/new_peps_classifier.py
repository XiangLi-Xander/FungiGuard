# -*- coding:utf-8 -*-
import torch
import numpy as np
import pandas as pd
from load_and_plot import *
from LSTMClassifier import LSTMClassifier
from LSTMATTClassifier import LSTMATTClassifier
from biLSTMClassifier import biLSTMClassifier
from biLSTMATTClassifier import biLSTMATTClassifier
from RFClassifier import *
import joblib
import argparse

def predict_with_model(model, sequences):
    """Predict using the provided PyTorch model."""
    predictions, probabilities = [], []
    for seq in sequences:
        inputs = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[:, 1].item()
        predictions.append(predicted.item())
        probabilities.append(probability)
    return predictions, probabilities


def load_model(model_type, model_path, input_size, hidden_size, output_size):
    """Load a PyTorch model from the given path."""
    model_classes = {
        'lstm': LSTMClassifier,
        'lstmatt': LSTMATTClassifier,
        'bilstm': biLSTMClassifier,
        'bilstmatt': biLSTMATTClassifier
    }
    model_class = model_classes.get(model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model_class(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"{model_type} model loaded from {model_path}")
    return model

def main(file_path):
    sequences = read_fasta(file_path)
    model_paths = {
        'lstm': '../models/lstm.pth',
        'lstmatt': '../models/lstmatt.pth',
        'bilstm': '../models/bilstm.pth',
        'bilstmatt': '../models/bilstmatt.pth',
        'rf': '../models/rf_model.pkl'
    }

    results = []
    for model_type, model_path in model_paths.items():
        if model_type == 'rf':
            model = load_rf_model(model_path)
            predictions, probabilities = predict_rf(model, sequences)
            print(f"RFClassifier model loaded from {model_path}")
        else:
            model = load_model(model_type, model_path, 100, 16, 2)
            predictions, probabilities = predict_with_model(model, sequences)
        results.extend({
            'Sequence': i + 1,
            'Model': model_type,
            'Predicted': pred,
            'Probability': prob if prob is not None else 'N/A'
        } for i, (pred, prob) in enumerate(zip(predictions, probabilities)))

    pd.DataFrame(results).to_excel('model_predictions.xlsx', index=False)
    print("Prediction results have been saved to model_predictions.xlsx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sequences using trained models")
    parser.add_argument('file_path', type=str, help='Path to the input FASTA file')
    args = parser.parse_args()
    main(args.file_path)