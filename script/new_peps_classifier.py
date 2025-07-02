# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import joblib
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_and_plot import read_fasta

# ===== 模型定义 =====
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc(out)

class LSTMATTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMATTClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention_layer(output), dim=1)
        attn_output = torch.sum(attn_weights * output, dim=1)
        out = self.dropout(attn_output)
        return self.fc(out)

class biLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(biLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = torch.cat((h_n[0], h_n[1]), dim=1)
        out = self.dropout(out)
        return self.fc(out)

class biLSTMATTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(biLSTMATTClassifier, self).__init__()
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
        output = output.permute(1, 0, 2)
        attn_output = torch.sum(attn_weights * output, dim=1)
        output = self.fc(attn_output)
        return output

# ===== 预测函数 =====
def predict_sklearn(model, sequences):
    X = np.array(sequences)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds, probs

def predict_pytorch(model, sequences):
    model.eval()
    with torch.no_grad():
        X = np.array(sequences)
        inputs = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)[:, 1]
        _, preds = torch.max(outputs, 1)
    return preds.numpy(), probs.numpy()

# ===== 主函数 =====
def main(fasta_file, model_path, output_path, max_len=100):
    ids, sequences = read_fasta(fasta_file, max_len=max_len)
    print(f"📄 Loaded {len(sequences)} sequences from {fasta_file}")

    model_path_lower = model_path.lower()

    if model_path_lower.endswith(".pkl"):
        model = joblib.load(model_path)
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

    # 保存结果
    df = pd.DataFrame({
        'ID': ids,
        'Prediction': preds,
        'Probability': probs
    })
    df.to_csv(output_path, index=False)
    print(f"📊 Results saved to {output_path}")

# ===== 命令行参数 =====
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict AFPs using one of 5 trained models")
    parser.add_argument("fasta", help="Input FASTA file path")
    parser.add_argument("--model", required=True, help="Path to the trained model file (.pkl or .pth)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file")
    parser.add_argument("--max_len", type=int, default=100, help="Max sequence length")
    args = parser.parse_args()

    main(args.fasta, args.model, args.output, args.max_len)
