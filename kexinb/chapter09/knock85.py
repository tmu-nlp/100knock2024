# task85. 双方向RNN・多層化

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import time
import numpy as np
from matplotlib import pyplot as plt
from knock81 import *
from gensim.models import KeyedVectors

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

# Define Bidirectional and Multilayer RNN model
class BiRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, emb_size, pad_idx, output_size, num_layers=2, pre_trained_embeddings=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        if pre_trained_embeddings is not None:
            self.emb.weight = nn.Parameter(torch.tensor(pre_trained_embeddings, dtype=torch.float32))
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 because of bidirectionality

    def forward(self, x):
        emb = self.emb(x)
        out, _ = self.rnn(emb)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
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

    plt.savefig("output/ch9/85.png")

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
    model = BiRNN(HIDDEN_SIZE, VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, pre_trained_embeddings=pretrained_embeddings)
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
(Batchsize = 8)
Epoch 1/20, Train Loss: 1.2014, Train Acc: 0.4117, Valid Loss: 1.2242, Valid Acc: 0.4070, Time: 17.88 sec
Epoch 2/20, Train Loss: 1.1786, Train Acc: 0.4200, Valid Loss: 1.2087, Valid Acc: 0.4078, Time: 15.35 sec
Epoch 3/20, Train Loss: 1.1697, Train Acc: 0.4168, Valid Loss: 1.2019, Valid Acc: 0.4070, Time: 13.16 sec
Epoch 4/20, Train Loss: 1.1580, Train Acc: 0.4268, Valid Loss: 1.1925, Valid Acc: 0.4168, Time: 13.89 sec
Epoch 5/20, Train Loss: 1.1501, Train Acc: 0.4269, Valid Loss: 1.1846, Valid Acc: 0.4258, Time: 13.74 sec
Epoch 6/20, Train Loss: 1.1299, Train Acc: 0.4506, Valid Loss: 1.1644, Valid Acc: 0.4558, Time: 13.98 sec
Epoch 7/20, Train Loss: 1.0073, Train Acc: 0.6124, Valid Loss: 1.0475, Valid Acc: 0.6087, Time: 16.14 sec
Epoch 8/20, Train Loss: 0.6945, Train Acc: 0.7650, Valid Loss: 0.7478, Valid Acc: 0.7429, Time: 15.63 sec
Epoch 9/20, Train Loss: 0.6480, Train Acc: 0.7748, Valid Loss: 0.7062, Valid Acc: 0.7511, Time: 16.52 sec
Epoch 10/20, Train Loss: 0.6156, Train Acc: 0.7807, Valid Loss: 0.6770, Valid Acc: 0.7526, Time: 20.35 sec
Epoch 11/20, Train Loss: 0.5985, Train Acc: 0.7820, Valid Loss: 0.6583, Valid Acc: 0.7526, Time: 18.41 sec
Epoch 12/20, Train Loss: 0.5698, Train Acc: 0.7899, Valid Loss: 0.6315, Valid Acc: 0.7631, Time: 17.22 sec
Epoch 13/20, Train Loss: 0.5784, Train Acc: 0.7830, Valid Loss: 0.6346, Valid Acc: 0.7609, Time: 18.82 sec
Epoch 14/20, Train Loss: 0.5374, Train Acc: 0.7957, Valid Loss: 0.5953, Valid Acc: 0.7714, Time: 18.62 sec
Epoch 15/20, Train Loss: 0.5100, Train Acc: 0.8065, Valid Loss: 0.5731, Valid Acc: 0.7826, Time: 14.34 sec
Epoch 16/20, Train Loss: 0.4845, Train Acc: 0.8141, Valid Loss: 0.5538, Valid Acc: 0.7856, Time: 13.10 sec
Epoch 17/20, Train Loss: 0.4797, Train Acc: 0.8161, Valid Loss: 0.5499, Valid Acc: 0.7901, Time: 12.93 sec
Epoch 18/20, Train Loss: 0.4510, Train Acc: 0.8254, Valid Loss: 0.5255, Valid Acc: 0.7946, Time: 14.60 sec
Epoch 19/20, Train Loss: 0.4315, Train Acc: 0.8342, Valid Loss: 0.5100, Valid Acc: 0.8051, Time: 21.25 sec
Epoch 20/20, Train Loss: 0.4097, Train Acc: 0.8447, Valid Loss: 0.5011, Valid Acc: 0.8013, Time: 20.49 sec
Train Accuracy: 0.845
Test Accuracy: 0.822
'''