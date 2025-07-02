# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import argparse
import joblib

from load_and_plot import read_fasta
from biLSTMATTClassifier import biLSTMATTClassifier
from biLSTMClassifier import biLSTMClassifier
from LSTMATTClassifier import LSTMATTClassifier
from LSTMClassifier import LSTMClassifier
from RFClassifier import load_rf_model

# ========= sklearn 预测 =========
def predict_sklearn(model, sequences):
    X = np.array(sequences)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds, probs

# ========= PyTorch 预测 =========
def predict_pytorch(model, sequences):
    model.eval()
    with torch.no_grad():
        X = np.array(sequences)
        inputs = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)[:, 1]
        _, preds = torch.max(outputs, 1)
    return preds.numpy(), probs.numpy()

# ========= 主函数 =========
def main(fasta_file, model_path, output_path, max_len=100):
    ids, sequences = read_fasta(fasta_file, max_len=max_len)
    print(f"📄 Loaded {len(sequences)} sequences from {fasta_file}")

    model_path_lower = model_path.lower()

    if model_path_lower.endswith(".pkl"):
        model = load_rf_model(model_path)
        print("✅ Loaded Random Forest model")
        preds, probs = predict_sklearn(model, sequences)

    elif model_path_lower.endswith(".pth"):
        input_size = max_len
        hidden_size = 16
        output_size = 2

        if "bilstmatt" in model_path_lower:
            model = biLSTMATTClassifier(input_size, hidden_size, output_size)
        elif "bilstm" in model_path_lower:
            model = biLSTMClassifier(input_size, hidden_size, output_size)
        elif "lstmatt" in model_path_lower:
            model = LSTMATTClassifier(input_size, hidden_size, output_size)
        elif "lstm" in model_path_lower:
            model = LSTMClassifier(input_size, hidden_size, output_size)
        else:
            raise ValueError("Unrecognized PyTorch model filename pattern.")

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"✅ Loaded PyTorch model: {model_path}")
        preds, probs = predict_pytorch(model, sequences)

    else:
        raise ValueError("Model file must be either .pkl or .pth")

    df = pd.DataFrame({
        'ID': ids,
        'Prediction': preds,
        'Probability': probs
    })
    df.to_csv(output_path, index=False)
    print(f"📊 Results saved to {output_path}")

# ========= 命令行入口 =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using 5 types of models")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("--model", required=True, help="Model file (.pkl or .pth)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file")
    parser.add_argument("--max_len", type=int, default=100, help="Max sequence length")
    args = parser.parse_args()

    main(args.fasta, args.model, args.output, args.max_len)