# 77

import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import time


# Function to save checkpoints
def save_checkpoint(epoch, W, train_loss, eval_loss, train_accuracy, eval_accuracy):
    checkpoint = {
        'epoch': epoch,
        'W': W,
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'train_accuracy': train_accuracy,
        'eval_accuracy': eval_accuracy
    }
    checkpoint_filename = f'checkpoint_epoch_{epoch}.pkl'
    with open(checkpoint_filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved for epoch {epoch}")


# Load the data

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
epochs = 10
batch_sizes = [1, 2, 4, 8, 16, 32, 64]  # Different batch sizes to compare

# Lists to store training time for each batch size
training_times = []

# Perform SGD with different batch sizes
for B in batch_sizes:
    print(f"Training with batch size: {B}")
    start_time = time.time()

    # Lists to store loss and accuracy for plotting
    train_losses = []
    eval_losses = []
    train_accuracies = []
    eval_accuracies = []

    # Number of mini-batches
    num_batches = len(X_train) // B

    for epoch in range(epochs):
        total_loss = 0

        # Shuffle training data
        shuffled_indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[shuffled_indices]
        Y_train_shuffled = Y_train[shuffled_indices]

        for batch in range(num_batches):
            start_idx = batch * B
            end_idx = start_idx + B

            # Extract mini-batch
            X_batch = X_train_shuffled[start_idx:end_idx]
            Y_batch = Y_train_shuffled[start_idx:end_idx]

            # Forward pass: compute prediction and loss for the mini-batch
            Y_hat_batch = softmax(np.dot(X_batch, W))
            loss_batch = cross_entropy_loss(Y_hat_batch, Y_batch)
            total_loss += loss_batch

            # Backward pass: compute gradients for the mini-batch
            gradients_batch = compute_gradients(X_batch, Y_batch, Y_hat_batch)

            # Update weights
            W -= learning_rate * gradients_batch

        # Average loss over the training data
        average_loss = total_loss / num_batches
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

        # Save checkpoint
        save_checkpoint(epoch + 1, W, average_loss, eval_loss, train_accuracy, eval_accuracy)

    end_time = time.time()
    training_time = end_time - start_time
    training_times.append(training_time)
    print(f"Training with batch size {B} completed in {training_time:.2f} seconds")
    print()

# Plotting training times for different batch sizes
plt.figure(figsize=(8, 5))
plt.plot(batch_sizes, training_times, marker='o', linestyle='-', color='b')
plt.title('Training Time vs Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Training Time (seconds)')
plt.grid(True)
plt.tight_layout()
plt.show()
