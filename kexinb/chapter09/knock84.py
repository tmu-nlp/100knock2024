# task84. 単語ベクトルの導入

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import time
from matplotlib import pyplot as plt
from gensim.models import KeyedVectors
from knock81 import *  

# Load pre-trained word vectors
def load_pretrained_vectors(filepath, word2id, embedding_dim):
    word_vectors = KeyedVectors.load_word2vec_format(filepath, binary=True)
    vocab_size = len(word2id)
    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))  # +1 for padding_idx

    for word, idx in word2id.items():
        if word in word_vectors:
            embedding_matrix[idx] = word_vectors[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    return embedding_matrix

# Define RNN model
class RNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, emb_size, pad_idx, output_size, pre_trained_embeddings=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        if pre_trained_embeddings is not None:
            self.emb.weight = nn.Parameter(torch.tensor(pre_trained_embeddings, dtype=torch.float32))
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity="tanh", batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.rnn(emb)
        out = self.fc(out[:, -1, :])
        return out

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
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
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

    plt.savefig("output/ch9/84.png")

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
    NUM_EPOCHS = 10
    # Load pre-trained embeddings
    PRETRAINED_EMBEDDING_PATH = 'data/GoogleNews-vectors-negative300.bin.gz'
    pretrained_embeddings = load_pretrained_vectors(PRETRAINED_EMBEDDING_PATH, word_ids.id_dict, EMB_SIZE)

    # Model, criterion, and optimizer
    model = RNN(HIDDEN_SIZE, VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, pre_trained_embeddings=pretrained_embeddings)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

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
Epoch 1/10, Train Loss: 1.2380, Train Acc: 0.3751, Valid Loss: 1.2492, Valid Acc: 0.3756, Time: 38.95 sec
Epoch 2/10, Train Loss: 1.2137, Train Acc: 0.3878, Valid Loss: 1.2323, Valid Acc: 0.3876, Time: 34.39 sec
Epoch 3/10, Train Loss: 1.2027, Train Acc: 0.3984, Valid Loss: 1.2241, Valid Acc: 0.4070, Time: 29.55 sec
Epoch 4/10, Train Loss: 1.1891, Train Acc: 0.4087, Valid Loss: 1.2144, Valid Acc: 0.4145, Time: 26.85 sec
Epoch 5/10, Train Loss: 1.1793, Train Acc: 0.4173, Valid Loss: 1.2071, Valid Acc: 0.4273, Time: 36.61 sec
Epoch 6/10, Train Loss: 1.1700, Train Acc: 0.4257, Valid Loss: 1.1991, Valid Acc: 0.4333, Time: 47.38 sec
Epoch 7/10, Train Loss: 1.1589, Train Acc: 0.4328, Valid Loss: 1.1891, Valid Acc: 0.4363, Time: 25.37 sec
Epoch 8/10, Train Loss: 1.1370, Train Acc: 0.4530, Valid Loss: 1.1689, Valid Acc: 0.4543, Time: 27.88 sec
Epoch 9/10, Train Loss: 0.8458, Train Acc: 0.7301, Valid Loss: 0.8792, Valid Acc: 0.7211, Time: 24.30 sec
Epoch 10/10, Train Loss: 0.7013, Train Acc: 0.7681, Valid Loss: 0.7487, Valid Acc: 0.7429, Time: 26.68 sec
Train Accuracy: 0.768
Test Accuracy: 0.749
'''