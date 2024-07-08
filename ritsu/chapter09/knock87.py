import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from knock81 import NewsDataset, load_data, create_word_to_id, pad_sequence
import os
import time

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_size, output_dim, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(in_channels=embedding_dim, 
                              out_channels=n_filters, 
                              kernel_size=filter_size,
                              padding=1)
        self.fc = nn.Linear(n_filters, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        conved = F.relu(self.conv(embedded))
        pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
        return self.fc(pooled)

def calculate_accuracy(predictions, labels):
    top_predictions = predictions.argmax(1, keepdim=True)
    correct = top_predictions.eq(labels.view_as(top_predictions)).sum()
    accuracy = correct.float() / labels.shape[0]
    return accuracy

def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        text, labels, _ = [x.to(device) for x in batch]
        
        optimizer.zero_grad()
        predictions = model(text)
        
        loss = criterion(predictions, labels)
        acc = calculate_accuracy(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in iterator:
            text, labels, _ = [x.to(device) for x in batch]
            
            predictions = model(text)
            
            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    base_path = os.path.join('..', 'chapter06')
    train_data = load_data(os.path.join(base_path, 'train.txt'))
    valid_data = load_data(os.path.join(base_path, 'valid.txt'))
    
    word_to_id = create_word_to_id(train_data)
    category_to_id = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    
    train_dataset = NewsDataset(train_data['TITLE'], train_data['CATEGORY'].map(category_to_id), word_to_id)
    valid_dataset = NewsDataset(valid_data['TITLE'], valid_data['CATEGORY'].map(category_to_id), word_to_id)
    
    BATCH_SIZE = 64
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_sequence)
    valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=pad_sequence)
    
    VOCAB_SIZE = len(word_to_id) + 1
    EMBEDDING_DIM = 300
    N_FILTERS = 100
    FILTER_SIZE = 3
    OUTPUT_DIM = 4
    PAD_IDX = 0
    
    model = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, N_FILTERS, FILTER_SIZE, OUTPUT_DIM, PAD_IDX).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    N_EPOCHS = 10
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'cnn-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

if __name__ == "__main__":
    main()

"""
Using device: cpu
Epoch: 01 | Epoch Time: 0m 1s
        Train Loss: 1.044 | Train Acc: 61.36%
         Val. Loss: 0.945 |  Val. Acc: 66.40%
Epoch: 02 | Epoch Time: 0m 1s
        Train Loss: 0.871 | Train Acc: 69.50%
         Val. Loss: 0.866 |  Val. Acc: 68.50%
Epoch: 03 | Epoch Time: 0m 1s
        Train Loss: 0.784 | Train Acc: 72.06%
         Val. Loss: 0.816 |  Val. Acc: 69.48%
Epoch: 04 | Epoch Time: 0m 1s
        Train Loss: 0.711 | Train Acc: 74.15%
         Val. Loss: 0.777 |  Val. Acc: 71.37%
Epoch: 05 | Epoch Time: 0m 1s
        Train Loss: 0.641 | Train Acc: 76.70%
         Val. Loss: 0.734 |  Val. Acc: 72.64%
Epoch: 06 | Epoch Time: 0m 1s
        Train Loss: 0.575 | Train Acc: 79.48%
         Val. Loss: 0.702 |  Val. Acc: 74.24%
Epoch: 07 | Epoch Time: 0m 1s
        Train Loss: 0.513 | Train Acc: 82.19%
         Val. Loss: 0.669 |  Val. Acc: 75.34%
Epoch: 08 | Epoch Time: 0m 1s
        Train Loss: 0.456 | Train Acc: 84.58%
         Val. Loss: 0.643 |  Val. Acc: 77.07%
Epoch: 09 | Epoch Time: 0m 1s
        Train Loss: 0.404 | Train Acc: 87.21%
         Val. Loss: 0.619 |  Val. Acc: 78.01%
Epoch: 10 | Epoch Time: 0m 1s
        Train Loss: 0.357 | Train Acc: 89.44%
         Val. Loss: 0.603 |  Val. Acc: 79.06%
"""