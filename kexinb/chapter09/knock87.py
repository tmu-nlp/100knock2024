# task87. 確率的勾配降下法によるCNNの学習

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

# Define CNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_size, pad_idx, num_classes, pre_trained_embeddings=None, kernel_size=3, hidden_size=50):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        if pre_trained_embeddings is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(pre_trained_embeddings, dtype=torch.float32))
        self.conv = nn.Conv2d(1, hidden_size, (kernel_size, emb_size), padding=(1, 0))  # Convolution layer
        self.fc = nn.Linear(hidden_size, num_classes)  # Fully connected layer

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv(x)).squeeze(3)  # Apply convolution and activation function
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # Apply max pooling
        x = self.fc(x)  # Fully connected layer
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

    plt.savefig("output/ch9/87.png")

# Main script
if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = len(set(word_ids.id_dict.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word_ids.id_dict.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 8
    NUM_EPOCHS = 20

    # Load pre-trained embeddings
    PRETRAINED_EMBEDDING_PATH = 'data/GoogleNews-vectors-negative300.bin.gz'
    pretrained_embeddings = load_pretrained_vectors(PRETRAINED_EMBEDDING_PATH, word_ids.id_dict, EMB_SIZE)

    # Model, criterion, and optimizer
    model = TextCNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, pre_trained_embeddings=pretrained_embeddings)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)  # Using SGD optimizer

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
Epoch 1/20, Train Loss: 1.1262, Train Acc: 0.6425, Valid Loss: 1.1526, Valid Acc: 0.6214, Time: 23.59 sec
Epoch 2/20, Train Loss: 1.0211, Train Acc: 0.7255, Valid Loss: 1.0607, Valid Acc: 0.7009, Time: 11.93 sec
Epoch 3/20, Train Loss: 0.9098, Train Acc: 0.7465, Valid Loss: 0.9577, Valid Acc: 0.7196, Time: 12.44 sec
Epoch 4/20, Train Loss: 0.8020, Train Acc: 0.7624, Valid Loss: 0.8568, Valid Acc: 0.7339, Time: 16.07 sec
Epoch 5/20, Train Loss: 0.7213, Train Acc: 0.7736, Valid Loss: 0.7807, Valid Acc: 0.7481, Time: 14.95 sec
Epoch 6/20, Train Loss: 0.6642, Train Acc: 0.7803, Valid Loss: 0.7265, Valid Acc: 0.7526, Time: 20.52 sec
Epoch 7/20, Train Loss: 0.6235, Train Acc: 0.7833, Valid Loss: 0.6876, Valid Acc: 0.7556, Time: 21.75 sec
Epoch 8/20, Train Loss: 0.5923, Train Acc: 0.7846, Valid Loss: 0.6576, Valid Acc: 0.7579, Time: 14.20 sec
Epoch 9/20, Train Loss: 0.5673, Train Acc: 0.7880, Valid Loss: 0.6333, Valid Acc: 0.7631, Time: 24.33 sec
Epoch 10/20, Train Loss: 0.5462, Train Acc: 0.7911, Valid Loss: 0.6136, Valid Acc: 0.7661, Time: 21.17 sec
Epoch 11/20, Train Loss: 0.5278, Train Acc: 0.7955, Valid Loss: 0.5963, Valid Acc: 0.7729, Time: 15.19 sec
Epoch 12/20, Train Loss: 0.5110, Train Acc: 0.8001, Valid Loss: 0.5803, Valid Acc: 0.7759, Time: 13.85 sec
Epoch 13/20, Train Loss: 0.4954, Train Acc: 0.8084, Valid Loss: 0.5648, Valid Acc: 0.7804, Time: 20.32 sec
Epoch 14/20, Train Loss: 0.4808, Train Acc: 0.8152, Valid Loss: 0.5514, Valid Acc: 0.7849, Time: 19.50 sec
Epoch 15/20, Train Loss: 0.4668, Train Acc: 0.8206, Valid Loss: 0.5387, Valid Acc: 0.7924, Time: 15.37 sec
Epoch 16/20, Train Loss: 0.4535, Train Acc: 0.8269, Valid Loss: 0.5257, Valid Acc: 0.7961, Time: 17.59 sec
Epoch 17/20, Train Loss: 0.4407, Train Acc: 0.8323, Valid Loss: 0.5150, Valid Acc: 0.8021, Time: 17.31 sec
Epoch 18/20, Train Loss: 0.4285, Train Acc: 0.8401, Valid Loss: 0.5021, Valid Acc: 0.8058, Time: 19.84 sec
Epoch 19/20, Train Loss: 0.4162, Train Acc: 0.8450, Valid Loss: 0.4918, Valid Acc: 0.8111, Time: 12.21 sec
Epoch 20/20, Train Loss: 0.4045, Train Acc: 0.8516, Valid Loss: 0.4812, Valid Acc: 0.8178, Time: 10.24 sec
Train Accuracy: 0.852
Test Accuracy: 0.834
'''