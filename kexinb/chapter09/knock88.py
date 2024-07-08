# task88. パラメータチューニング

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import time
import numpy as np
from matplotlib import pyplot as plt
from knock81 import *
from gensim.models import KeyedVectors

# Load pre-trained word vectors
def load_pretrained_vectors(filepath, word2id, embedding_dim):
    word_vectors = KeyedVectors.load_word2vec_format(filepath, binary=True)  # Load word vectors
    vocab_size = len(word2id)
    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))  # +1 for padding_idx

    for word, idx in word2id.items():
        if word in word_vectors:
            embedding_matrix[idx] = word_vectors[word]  # Assign pre-trained vectors
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))  # Random initialization for missing words
    
    return embedding_matrix

# Define enhanced CNN model with dropout and regularization
class EnhancedTextCNN(nn.Module):
    def __init__(self, vocab_size, emb_size, pad_idx, num_classes, pre_trained_embeddings=None, 
                 kernel_size=3, num_filters=100, hidden_size=50, dropout_rate=0.6):
        super(EnhancedTextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        if pre_trained_embeddings is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(pre_trained_embeddings, dtype=torch.float32))
        
        self.conv = nn.Conv2d(1, num_filters, (kernel_size, emb_size), padding=(1, 0))  # Adjusted Convolution layer
        self.dropout = nn.Dropout(dropout_rate)  # Increased Dropout
        self.fc = nn.Linear(num_filters, hidden_size)  # Adjusted Fully connected layer
        self.out = nn.Linear(hidden_size, num_classes)  # Output layer

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv(x)).squeeze(3)  # Apply convolution and activation function
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # Apply max pooling
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc(x))  # Apply fully connected layer
        x = self.out(x)  # Output layer
        return x

# Define custom dataset
class NewsDataset(Dataset):
    def __init__(self, x, y, tokenizer):
        self.x = x
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        text = self.x[idx]
        inputs = self.tokenizer(text)
        return {
            'inputs': torch.tensor(inputs, dtype=torch.int64),
            'labels': torch.tensor(self.y[idx], dtype=torch.int64)
        }

# Custom collate function to pad sequences
def collate_fn(batch):
    inputs = [item['inputs'] for item in batch]
    labels = [item['labels'] for item in batch]
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)  # Pad sequences
    labels = torch.stack(labels)
    return {'inputs': inputs_padded, 'labels': labels}

# Function to calculate loss and accuracy
def calc_loss_acc(model, dataset, device=None, criterion=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs = data["inputs"].to(device)
            labels = data["labels"].to(device)
            outputs = model(inputs)

            if criterion is not None:
                loss += criterion(outputs, labels).item()

            pred = torch.argmax(outputs, dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return loss / len(dataloader), correct / total

# Function to train the model
def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device=None):
    model.to(device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, collate_fn=collate_fn)

    log_train = []
    log_valid = []

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        for data in dataloader_train:
            optimizer.zero_grad()

            inputs = data["inputs"].to(device)
            labels = data["labels"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        end_time = time.time()

        loss_train, acc_train = calc_loss_acc(model, dataset_train, device, criterion)
        loss_valid, acc_valid = calc_loss_acc(model, dataset_valid, device, criterion)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': loss_train,
            'acc_train': acc_train,
            'loss_valid': loss_valid,
            'acc_valid': acc_valid
        }, f'output/ch9/checkpoint{epoch + 1}.pt')

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.4f}, '
              f'Valid Loss: {loss_valid:.4f}, Valid Acc: {acc_valid:.4f}, '
              f'Time: {end_time - start_time:.2f} sec')

    return {"train": log_train, "valid": log_valid}

# Function to visualize training and validation logs
def visualize_logs(log):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(np.array(log['train']).T[0], label='Train')
    ax[0].plot(np.array(log['valid']).T[0], label='Valid')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(np.array(log['train']).T[1], label='Train')
    ax[1].plot(np.array(log['valid']).T[1], label='Valid')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.savefig("output/ch9/88.png")

# Main script
if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = len(set(word_ids.id_dict.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word_ids.id_dict.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64  # Increased batch size for better performance
    NUM_EPOCHS = 20
    NUM_FILTERS = 128  # Increased number of filters
    DROPOUT_RATE = 0.6  # Increased dropout rate

    # Load pre-trained embeddings
    PRETRAINED_EMBEDDING_PATH = 'data/GoogleNews-vectors-negative300.bin.gz'
    pretrained_embeddings = load_pretrained_vectors(PRETRAINED_EMBEDDING_PATH, word_ids.id_dict, EMB_SIZE)

    # Model, criterion, and optimizer
    model = EnhancedTextCNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, pre_trained_embeddings=pretrained_embeddings, num_filters=NUM_FILTERS, hidden_size=HIDDEN_SIZE, dropout_rate=DROPOUT_RATE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train and log the model
    log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Visualize training logs
    visualize_logs(log)

    # Calculate and print final accuracy on train and test sets
    _, acc_train = calc_loss_acc(model, dataset_train, device='cuda' if torch.cuda.is_available() else 'cpu', criterion=criterion)
    _, acc_test = calc_loss_acc(model, dataset_test, device='cuda' if torch.cuda.is_available() else 'cpu', criterion=criterion)
    print(f'Train Accuracy: {acc_train:.3f}')
    print(f'Test Accuracy: {acc_test:.3f}')


'''
Epoch 1/20, Train Loss: 0.3080, Train Acc: 0.8975, Valid Loss: 0.4097, Valid Acc: 0.8606, Time: 7.22 sec
Epoch 2/20, Train Loss: 0.1291, Train Acc: 0.9628, Valid Loss: 0.3040, Valid Acc: 0.9003, Time: 6.90 sec
Epoch 3/20, Train Loss: 0.0669, Train Acc: 0.9813, Valid Loss: 0.3046, Valid Acc: 0.9018, Time: 7.17 sec
Epoch 4/20, Train Loss: 0.0345, Train Acc: 0.9910, Valid Loss: 0.3357, Valid Acc: 0.9070, Time: 6.96 sec
Epoch 5/20, Train Loss: 0.0193, Train Acc: 0.9948, Valid Loss: 0.3803, Valid Acc: 0.9010, Time: 7.21 sec
Epoch 6/20, Train Loss: 0.0121, Train Acc: 0.9966, Valid Loss: 0.4251, Valid Acc: 0.9048, Time: 7.15 sec
Epoch 7/20, Train Loss: 0.0092, Train Acc: 0.9979, Valid Loss: 0.4340, Valid Acc: 0.9093, Time: 7.23 sec
Epoch 8/20, Train Loss: 0.0069, Train Acc: 0.9985, Valid Loss: 0.4507, Valid Acc: 0.9063, Time: 8.97 sec
Epoch 9/20, Train Loss: 0.0062, Train Acc: 0.9983, Valid Loss: 0.5065, Valid Acc: 0.9040, Time: 6.99 sec
Epoch 10/20, Train Loss: 0.0044, Train Acc: 0.9989, Valid Loss: 0.5372, Valid Acc: 0.9055, Time: 7.05 sec
Epoch 11/20, Train Loss: 0.0042, Train Acc: 0.9988, Valid Loss: 0.5437, Valid Acc: 0.9078, Time: 7.07 sec
Epoch 12/20, Train Loss: 0.0046, Train Acc: 0.9990, Valid Loss: 0.5768, Valid Acc: 0.9078, Time: 7.02 sec
Epoch 13/20, Train Loss: 0.0039, Train Acc: 0.9991, Valid Loss: 0.6279, Valid Acc: 0.9078, Time: 7.15 sec
Epoch 14/20, Train Loss: 0.0037, Train Acc: 0.9990, Valid Loss: 0.6061, Valid Acc: 0.9055, Time: 7.11 sec
Epoch 15/20, Train Loss: 0.0052, Train Acc: 0.9988, Valid Loss: 0.6484, Valid Acc: 0.9078, Time: 6.93 sec
Epoch 16/20, Train Loss: 0.0029, Train Acc: 0.9990, Valid Loss: 0.6409, Valid Acc: 0.9078, Time: 7.00 sec
Epoch 17/20, Train Loss: 0.0026, Train Acc: 0.9991, Valid Loss: 0.6434, Valid Acc: 0.9093, Time: 7.90 sec
Epoch 18/20, Train Loss: 0.0027, Train Acc: 0.9991, Valid Loss: 0.6628, Valid Acc: 0.9070, Time: 9.04 sec
Epoch 19/20, Train Loss: 0.0032, Train Acc: 0.9988, Valid Loss: 0.6923, Valid Acc: 0.9033, Time: 13.97 sec
Epoch 20/20, Train Loss: 0.0024, Train Acc: 0.9991, Valid Loss: 0.6958, Valid Acc: 0.9063, Time: 9.49 sec
Train Accuracy: 0.999
Test Accuracy: 0.906
'''