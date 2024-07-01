# 71

import numpy as np
import pickle
import torch

# Try loading with pickle first
try:
    with open('/content/drive/MyDrive/NLP/X_train.pt', 'rb') as f:
        X_train = pickle.load(f)
    with open('/content/drive/MyDrive/NLP/y_train.pt', 'rb') as f:
        Y_train = pickle.load(f)
except pickle.UnpicklingError:
    # If pickle loading fails, try torch loading
    X_train = torch.load('/content/drive/MyDrive/NLP/X_train.pt').numpy()
    Y_train = torch.load('/content/drive/MyDrive/NLP/y_train.pt').numpy()

# Define the softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtracting np.max(z) for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Initialize the weight matrix W randomly
d = 300  # Dimension of word vectors
L = 4    # Number of categories
W = np.random.randn(d, L)

# Compute y_hat_1
x1 = X_train[0].reshape(1, -1)  # x1 as a row vector
y_hat_1 = softmax(np.dot(x1, W))
print("y_hat_1:", y_hat_1)

# Compute Y_hat for the first four instances
X_1_4 = X_train[:4]
Y_hat = softmax(np.dot(X_1_4, W))
print("Y_hat:", Y_hat)
