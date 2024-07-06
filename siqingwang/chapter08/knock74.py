# 74

import numpy as np
import pickle
import torch

# Load the data
try:
    with open('X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('Y_train.pkl', 'rb') as f:
        Y_train = pickle.load(f)
    with open('X_eval.pkl', 'rb') as f:
        X_eval = pickle.load(f)
    with open('Y_eval.pkl', 'rb') as f:
        Y_eval = pickle.load(f)
except (pickle.UnpicklingError, FileNotFoundError):
    # If pickle loading fails, try torch loading
    X_train = torch.load('/content/drive/MyDrive/NLP/X_train.pt').numpy()
    Y_train = torch.load('/content/drive/MyDrive/NLP/y_train.pt').numpy()
    X_eval = torch.load('/content/drive/MyDrive/NLP/X_test.pt').numpy()
    Y_eval = torch.load('/content/drive/MyDrive/NLP/y_test.pt').numpy()

# Define the softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtracting np.max(z) for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Predict the classes
def predict(X, W):
    logits = np.dot(X, W)
    probs = softmax(logits)
    predictions = np.argmax(probs, axis=1)
    return predictions

# Calculate accuracy
def calculate_accuracy(predictions, labels):
    return np.mean(predictions == labels)

# Predict and calculate accuracy on the training data
train_predictions = predict(X_train, W)
train_accuracy = calculate_accuracy(train_predictions, Y_train)
print(f'Training Accuracy: {train_accuracy:.4f}')

# Predict and calculate accuracy on the evaluation data
eval_predictions = predict(X_eval, W)
eval_accuracy = calculate_accuracy(eval_predictions, Y_eval)
print(f'Evaluation Accuracy: {eval_accuracy:.4f}')
