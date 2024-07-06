# 79


import torch
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt


# Function to save checkpoints
def save_checkpoint(epoch, model, optimizer, train_loss, eval_loss, train_accuracy, eval_accuracy):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'train_accuracy': train_accuracy,
        'eval_accuracy': eval_accuracy
    }
    checkpoint_filename = f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_filename)
    print(f"Checkpoint saved for epoch {epoch}")


# Load the data

X_train = torch.load('/content/drive/MyDrive/NLP/X_train.pt').numpy()
Y_train = torch.load('/content/drive/MyDrive/NLP/y_train.pt').numpy()
X_eval = torch.load('/content/drive/MyDrive/NLP/X_test.pt').numpy()
Y_eval = torch.load('/content/drive/MyDrive/NLP/y_test.pt').numpy()

# Convert to PyTorch tensors and move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
Y_train = torch.tensor(Y_train, device=device, dtype=torch.long)
X_eval = torch.tensor(X_eval, device=device, dtype=torch.float32)
Y_eval = torch.tensor(Y_eval, device=device, dtype=torch.long)


# Define the neural network model
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


# Instantiate the model
input_size = 300  # Dimension of word vectors
hidden_size1 = 128
hidden_size2 = 64
output_size = 4  # Number of categories
model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size).to(device)

# Set hyperparameters
learning_rate = 0.01
epochs = 10
batch_sizes = [1, 2, 4, 8, 16, 32, 64]  # Different batch sizes to compare


# Define the softmax function
def softmax(z):
    exp_z = torch.exp(z - torch.max(z, dim=1, keepdim=True)[0])  # Subtracting max for numerical stability
    return exp_z / torch.sum(exp_z, dim=1, keepdim=True)


# Define the cross-entropy loss function
def cross_entropy_loss(Y_hat, Y):
    m = Y.shape[0]
    correct_logprobs = -torch.log(Y_hat[torch.arange(m), Y])
    loss = torch.sum(correct_logprobs) / m
    return loss.item()  # Return Python number


# Compute gradients for model parameters
def compute_gradients(loss, model):
    model.zero_grad()  # Clear gradients before backward pass
    loss.backward()  # Backpropagation
    # Gradient clipping can be applied if necessary
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
    return


# Predict the classes
def predict(model, X):
    logits = model(X)
    probs = softmax(logits)
    predictions = torch.argmax(probs, dim=1)
    return predictions


# Calculate accuracy
def calculate_accuracy(predictions, labels):
    return torch.mean((predictions == labels).float()).item()


# Training loop
def train(model, X_train, Y_train, X_eval, Y_eval, optimizer, criterion, epochs, batch_size):
    train_losses = []
    eval_losses = []
    train_accuracies = []
    eval_accuracies = []

    num_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        total_loss = 0

        # Shuffle training data
        shuffled_indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train[shuffled_indices]
        Y_train_shuffled = Y_train[shuffled_indices]

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size

            # Extract mini-batch
            X_batch = X_train_shuffled[start_idx:end_idx]
            Y_batch = Y_train_shuffled[start_idx:end_idx]

            # Forward pass
            Y_hat_batch = model(X_batch)
            loss = criterion(Y_hat_batch, Y_batch)
            total_loss += loss.item()

            # Backward pass
            compute_gradients(loss, model)

            # Update weights
            optimizer.step()

        # Average loss over the training data
        average_loss = total_loss / num_batches
        train_losses.append(average_loss)

        # Calculate accuracy on the training data
        train_predictions = predict(model, X_train)
        train_accuracy = calculate_accuracy(train_predictions, Y_train)
        train_accuracies.append(train_accuracy)

        # Calculate loss and accuracy on the evaluation data
        eval_predictions = predict(model, X_eval)
        eval_loss = criterion(model(X_eval), Y_eval)
        eval_losses.append(eval_loss.item())
        eval_accuracy = calculate_accuracy(eval_predictions, Y_eval)
        eval_accuracies.append(eval_accuracy)

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}')

        # Save checkpoint
        save_checkpoint(epoch + 1, model, optimizer, average_loss, eval_loss.item(), train_accuracy, eval_accuracy)

    return train_losses, eval_losses, train_accuracies, eval_accuracies


# Optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Lists to store training time for each batch size
training_times = []

# Perform training with different batch sizes
for batch_size in batch_sizes:
    print(f"Training with batch size: {batch_size}")
    start_time = time.time()

    # Train the model
    train_losses, eval_losses, train_accuracies, eval_accuracies = train(model, X_train, Y_train, X_eval, Y_eval,
                                                                         optimizer, criterion, epochs, batch_size)

    end_time = time.time()
    training_time = end_time - start_time
    training_times.append(training_time)
    print(f"Training with batch size {batch_size} completed in {training_time:.2f} seconds")
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
