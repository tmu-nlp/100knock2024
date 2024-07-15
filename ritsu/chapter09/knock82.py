import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from knock81 import RNN, NewsDataset, load_data, create_word_to_id, pad_sequence
import os
import time

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        texts, labels, lengths = batch
        
        optimizer.zero_grad()
        predictions = model(texts, lengths)
        
        loss = criterion(predictions, labels)
        acc = calculate_accuracy(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in iterator:
            texts, labels, lengths = batch
            
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
    
    VOCAB_SIZE = len(word_to_id) + 1  # +1 for padding
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 50
    OUTPUT_DIM = 4
    PAD_IDX = 0
    
    model = RNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX)
    
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    criterion = nn.CrossEntropyLoss()
    
    N_EPOCHS = 10
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
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
Epoch: 01 | Epoch Time: 0m 2s
        Train Loss: 1.140 | Train Acc: 50.33%
        Valid Loss: 1.094 | Valid Acc: 53.57%
Epoch: 02 | Epoch Time: 0m 2s
        Train Loss: 1.021 | Train Acc: 59.62%
        Valid Loss: 1.017 | Valid Acc: 61.08%
Epoch: 03 | Epoch Time: 0m 2s
        Train Loss: 0.886 | Train Acc: 68.01%
        Valid Loss: 0.908 | Valid Acc: 67.27%
Epoch: 04 | Epoch Time: 0m 2s
        Train Loss: 0.772 | Train Acc: 73.24%
        Valid Loss: 0.870 | Valid Acc: 71.09%
Epoch: 05 | Epoch Time: 0m 2s
        Train Loss: 0.688 | Train Acc: 76.34%
        Valid Loss: 0.839 | Valid Acc: 71.03%
Epoch: 06 | Epoch Time: 0m 2s
        Train Loss: 0.633 | Train Acc: 78.12%
        Valid Loss: 0.866 | Valid Acc: 70.93%
Epoch: 07 | Epoch Time: 0m 2s
        Train Loss: 0.567 | Train Acc: 80.39%
        Valid Loss: 0.839 | Valid Acc: 72.59%
Epoch: 08 | Epoch Time: 0m 2s
        Train Loss: 0.512 | Train Acc: 82.66%
        Valid Loss: 0.810 | Valid Acc: 72.28%
Epoch: 09 | Epoch Time: 0m 2s
        Train Loss: 0.471 | Train Acc: 83.61%
        Valid Loss: 0.839 | Valid Acc: 72.47%
Epoch: 10 | Epoch Time: 0m 2s
        Train Loss: 0.443 | Train Acc: 84.63%
        Valid Loss: 0.852 | Valid Acc: 73.96%
"""