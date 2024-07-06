# 75

import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt

X_train = torch.load('/content/drive/MyDrive/NLP/X_train.pt').numpy()
Y_train = torch.load('/content/drive/MyDrive/NLP/y_train.pt').numpy()
X_eval = torch.load('/content/drive/MyDrive/NLP/X_test.pt').numpy()
Y_eval = torch.load('/content/drive/MyDrive/NLP/y_test.pt').numpy()


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


# Predict the classes
def predict(X, W):
    logits = np.dot(X, W)
    probs = softmax(logits)
    predictions = np.argmax(probs, axis=1)
    return predictions


# Calculate accuracy
def calculate_accuracy(predictions, labels):
    return np.mean(predictions == labels)


# Initialize the weight matrix W randomly
d = 300  # Dimension of word vectors
L = 4  # Number of categories
W = np.random.randn(d, L)

# Set hyperparameters
learning_rate = 0.01
epochs = 100

# Lists to store loss and accuracy for plotting
train_losses = []
eval_losses = []
train_accuracies = []
eval_accuracies = []

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
    train_losses.append(average_loss)

    # Calculate accuracy on the training data
    train_predictions = predict(X_train, W)
    train_accuracy = calculate_accuracy(train_predictions, Y_train)
    train_accuracies.append(train_accuracy)

    # Calculate loss and accuracy on the evaluation data
    eval_predictions = predict(X_eval, W)
    eval_loss = cross_entropy_loss(softmax(np.dot(X_eval, W)), Y_eval)
    eval_losses.append(eval_loss)
    eval_accuracy = calculate_accuracy(eval_predictions, Y_eval)
    eval_accuracies.append(eval_accuracy)

    print(
        f'Epoch {epoch + 1}/{epochs}, Train Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}')

# Plotting the loss and accuracy
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, eval_losses, label='Evaluation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, eval_accuracies, label='Evaluation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
