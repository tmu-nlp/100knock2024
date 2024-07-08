import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from knock81 import RNN, NewsDataset, load_data, create_word_to_id, pad_sequence
import os
import time

def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        texts, labels, lengths = [x.to(device) for x in batch]
        
        optimizer.zero_grad()
        predictions = model(texts, lengths)
        
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
            texts, labels, lengths = [x.to(device) for x in batch]
            
            predictions = model(texts, lengths)
            
            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def calculate_accuracy(predictions, labels):
    top_predictions = predictions.argmax(1, keepdim=True)
    correct = top_predictions.eq(labels.view_as(top_predictions)).sum()
    accuracy = correct.float() / labels.shape[0]
    return accuracy

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    # GPUが利用可能な場合はGPUを使用、そうでない場合はCPUを使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    base_path = os.path.join('..', 'chapter06')
    train_data = load_data(os.path.join(base_path, 'train.txt'))
    valid_data = load_data(os.path.join(base_path, 'valid.txt'))
    
    word_to_id = create_word_to_id(train_data)
    category_to_id = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    
    train_dataset = NewsDataset(train_data['TITLE'], train_data['CATEGORY'].map(category_to_id), word_to_id)
    valid_dataset = NewsDataset(valid_data['TITLE'], valid_data['CATEGORY'].map(category_to_id), word_to_id)
    
    BATCH_SIZE = 64  # ミニバッチサイズ
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_sequence)
    valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=pad_sequence)
    
    VOCAB_SIZE = len(word_to_id) + 1  # +1 for padding
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 50
    OUTPUT_DIM = 4
    PAD_IDX = 0
    
    model = RNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
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
            torch.save(model.state_dict(), 'best-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')

if __name__ == "__main__":
    main()

"""
Using device: cpu
Epoch: 01 | Epoch Time: 0m 1s
        Train Loss: 1.126 | Train Acc: 52.47%
        Valid Loss: 1.069 | Valid Acc: 56.59%
Epoch: 02 | Epoch Time: 0m 1s
        Train Loss: 0.989 | Train Acc: 61.98%
        Valid Loss: 0.948 | Valid Acc: 66.55%
Epoch: 03 | Epoch Time: 0m 1s
        Train Loss: 0.844 | Train Acc: 70.34%
        Valid Loss: 0.883 | Valid Acc: 69.87%
Epoch: 04 | Epoch Time: 0m 1s
        Train Loss: 0.743 | Train Acc: 74.38%
        Valid Loss: 0.858 | Valid Acc: 71.92%
Epoch: 05 | Epoch Time: 0m 1s
        Train Loss: 0.661 | Train Acc: 77.45%
        Valid Loss: 0.841 | Valid Acc: 72.27%
Epoch: 06 | Epoch Time: 0m 1s
        Train Loss: 0.600 | Train Acc: 79.13%
        Valid Loss: 0.888 | Valid Acc: 72.37%
Epoch: 07 | Epoch Time: 0m 2s
        Train Loss: 0.550 | Train Acc: 81.02%
        Valid Loss: 0.845 | Valid Acc: 72.33%
Epoch: 08 | Epoch Time: 0m 1s
        Train Loss: 0.520 | Train Acc: 82.02%
        Valid Loss: 0.850 | Valid Acc: 74.69%
Epoch: 09 | Epoch Time: 0m 1s
        Train Loss: 0.476 | Train Acc: 83.92%
        Valid Loss: 0.864 | Valid Acc: 72.80%
Epoch: 10 | Epoch Time: 0m 1s
        Train Loss: 0.443 | Train Acc: 84.44%
        Valid Loss: 0.928 | Valid Acc: 73.37%
"""