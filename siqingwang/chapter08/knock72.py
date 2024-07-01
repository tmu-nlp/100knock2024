# 72

import numpy as np
import pickle
import torch

# Load the data
try:
    with open('X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('Y_train.pkl', 'rb') as f:
        Y_train = pickle.load(f)
except (pickle.UnpicklingError, FileNotFoundError):
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

# Compute softmax output for x1
x1 = X_train[0].reshape(1, -1)  # x1 as a row vector
y_hat_1 = softmax(np.dot(x1, W))

# Compute cross-entropy loss for a single sample
y1 = Y_train[0]
l1 = -np.log(y_hat_1[0, y1])
print("Loss for x1:", l1)

# Compute softmax output for the first four instances
X_1_4 = X_train[:4]
Y_1_4 = Y_train[:4]
Y_hat = softmax(np.dot(X_1_4, W))

# Compute average cross-entropy loss for the first four instances
losses = -np.log(Y_hat[np.arange(4), Y_1_4])
average_loss = np.mean(losses)
print("Average loss for x1, x2, x3, x4:", average_loss)

# Compute gradients for W
def compute_gradients(X, Y, Y_hat, W):
    m = X.shape[0]
    Y_one_hot = np.zeros_like(Y_hat)
    Y_one_hot[np.arange(m), Y] = 1
    gradients = np.dot(X.T, (Y_hat - Y_one_hot)) / m
    return gradients

# Compute gradients for x1
gradients_x1 = compute_gradients(x1, np.array([y1]), y_hat_1, W)
print("Gradients for W with respect to x1:", gradients_x1)

# Compute gradients for x1, x2, x3, x4
gradients_X_1_4 = compute_gradients(X_1_4, Y_1_4, Y_hat, W)
print("Gradients for W with respect to x1, x2, x3, x4:", gradients_X_1_4)
