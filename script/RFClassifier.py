# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from load_and_plot import *

def train_model(X_train, y_train):
    """Train RandomForest model with GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=3)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    """Evaluate the model performance on test data."""
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    probabilities = model.predict_proba(X)[:, 1]  # Probability of the positive class
    return y, predictions, probabilities, accuracy

def predict_rf(model, sequences):
    """Predict using the RandomForest model."""
    predictions = []
    probabilities = []
    for seq in sequences:
        seq = np.array(seq).reshape(1, -1)  # Reshape for a single sample
        probas = model.predict_proba(seq)
        predicted = model.predict(seq)[0]
        probability = probas[0, 1]  # Probability of the positive class
        predictions.append(predicted)
        probabilities.append(probability)
    return predictions, probabilities

def save_model(model, path):
    """Save the trained model to a file."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")

# Read and process data
data1 = pd.read_excel('../data/no.xlsx')
data2 = pd.read_excel('../data/antifu.xlsx')
maxseqlen = 100
seq2num(data1, data2, maxseqlen)
inputseq = 'seq2num.csv'
X_train, y_train, X_test, y_test = data_load(inputseq)

# Train and save the model
model = train_model(X_train, y_train)
save_model(model.best_estimator_, '../models/rf_model.pkl')

# Evaluate the model
y_true, y_pred, probabilities, accuracy = evaluate_model(model, X_test, y_test)
evaluate_results = pd.DataFrame({
    'labels': y_true,
    'predicted': y_pred,
    'probability': probabilities
})
evaluate_results.to_csv('../data/RandomForest_evaluation_results.csv', index=False)
print(f"Evaluation results saved to RandomForest_evaluation_results.csv with accuracy {accuracy:.4f}")

# Optional plotting
# plot_confusion(y_true, y_pred, "fig/RandomForest_confusion.png")
# plot_auc_curve(y_true, probabilities, "fig/RandomForest_AUC.png")