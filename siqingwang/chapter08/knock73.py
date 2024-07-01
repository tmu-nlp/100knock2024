# 73

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


# Define the cross-entropy loss function
def cross_entropy_loss(Y_hat, Y):
    m = Y.shape[0]
    correct_logprobs = -np.log(Y_hat[np.arange(m), Y])
    loss = np.sum(correct_logprobs) / m
    return loss


# Compute gradients for W
def compute_gradients(X, Y, Y_hat):
    m = X.shape[0]
    Y_one_hot = np.zeros_like(Y_hat)
    Y_one_hot[np.arange(m), Y] = 1
    gradients = np.dot(X.T, (Y_hat - Y_one_hot)) / m
    return gradients


# Initialize the weight matrix W randomly
d = 300  # Dimension of word vectors
L = 4  # Number of categories
W = np.random.randn(d, L)

# Set hyperparameters
learning_rate = 0.01
epochs = 100

# Perform SGD
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X_train)):
        x_i = X_train[i].reshape(1, -1)  # ith sample as a row vector
        y_i = Y_train[i]

        # Forward pass: compute prediction and loss
        y_hat_i = softmax(np.dot(x_i, W))
        loss_i = cross_entropy_loss(y_hat_i, np.array([y_i]))
        total_loss += loss_i

        # Backward pass: compute gradients
        gradients = compute_gradients(x_i, np.array([y_i]), y_hat_i)

        # Update weights
        W -= learning_rate * gradients

    # Average loss over the training data
    average_loss = total_loss / len(X_train)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}')

print("Training completed.")
