# task 86.畳み込みニューラルネットワーク (CNN)

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

    plt.savefig("output/ch9/86.png")

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
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Train and log the model
    log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Visualize training logs
    visualize_logs(log)

    # Calculate and print final accuracy on train and test sets
    _, acc_train = calc_loss_acc(model, dataset_train, device='cuda' 
                                 if torch.cuda.is_available() else 'cpu', criterion=criterion)
    _, acc_test = calc_loss_acc(model, dataset_test, device='cuda' 
                                if torch.cuda.is_available() else 'cpu', criterion=criterion)
    print(f'Train Accuracy: {acc_train:.3f}')
    print(f'Test Accuracy: {acc_test:.3f}')

'''
Epoch 1/20, Train Loss: 1.1242, Train Acc: 0.6878, Valid Loss: 1.1507, Valid Acc: 0.6627, Time: 13.96 sec
Epoch 2/20, Train Loss: 1.0349, Train Acc: 0.7046, Valid Loss: 1.0720, Valid Acc: 0.6874, Time: 13.47 sec
Epoch 3/20, Train Loss: 0.9319, Train Acc: 0.7321, Valid Loss: 0.9752, Valid Acc: 0.7114, Time: 14.60 sec
Epoch 4/20, Train Loss: 0.8247, Train Acc: 0.7583, Valid Loss: 0.8746, Valid Acc: 0.7301, Time: 13.38 sec
Epoch 5/20, Train Loss: 0.7391, Train Acc: 0.7710, Valid Loss: 0.7932, Valid Acc: 0.7451, Time: 17.86 sec
Epoch 6/20, Train Loss: 0.6783, Train Acc: 0.7789, Valid Loss: 0.7344, Valid Acc: 0.7534, Time: 13.41 sec
Epoch 7/20, Train Loss: 0.6340, Train Acc: 0.7832, Valid Loss: 0.6922, Valid Acc: 0.7571, Time: 15.37 sec
Epoch 8/20, Train Loss: 0.6011, Train Acc: 0.7852, Valid Loss: 0.6600, Valid Acc: 0.7594, Time: 14.03 sec
Epoch 9/20, Train Loss: 0.5754, Train Acc: 0.7877, Valid Loss: 0.6344, Valid Acc: 0.7631, Time: 19.68 sec
Epoch 10/20, Train Loss: 0.5537, Train Acc: 0.7913, Valid Loss: 0.6142, Valid Acc: 0.7699, Time: 12.81 sec
Epoch 11/20, Train Loss: 0.5351, Train Acc: 0.7942, Valid Loss: 0.5971, Valid Acc: 0.7721, Time: 42.84 sec
Epoch 12/20, Train Loss: 0.5190, Train Acc: 0.7973, Valid Loss: 0.5815, Valid Acc: 0.7759, Time: 23.93 sec
Epoch 13/20, Train Loss: 0.5042, Train Acc: 0.8018, Valid Loss: 0.5678, Valid Acc: 0.7766, Time: 21.19 sec
Epoch 14/20, Train Loss: 0.4906, Train Acc: 0.8077, Valid Loss: 0.5548, Valid Acc: 0.7819, Time: 14.45 sec
Epoch 15/20, Train Loss: 0.4776, Train Acc: 0.8130, Valid Loss: 0.5429, Valid Acc: 0.7864, Time: 15.43 sec
Epoch 16/20, Train Loss: 0.4654, Train Acc: 0.8174, Valid Loss: 0.5316, Valid Acc: 0.7894, Time: 14.73 sec
Epoch 17/20, Train Loss: 0.4529, Train Acc: 0.8246, Valid Loss: 0.5201, Valid Acc: 0.7961, Time: 17.56 sec
Epoch 18/20, Train Loss: 0.4407, Train Acc: 0.8319, Valid Loss: 0.5089, Valid Acc: 0.8036, Time: 14.22 sec
Epoch 19/20, Train Loss: 0.4286, Train Acc: 0.8400, Valid Loss: 0.4976, Valid Acc: 0.8088, Time: 14.94 sec
Epoch 20/20, Train Loss: 0.4171, Train Acc: 0.8471, Valid Loss: 0.4862, Valid Acc: 0.8156, Time: 16.06 sec
Train Accuracy: 0.847
Test Accuracy: 0.832
'''