import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from knock81 import NewsDataset, load_data, create_word_to_id, pad_sequence
import os
import time
import gensim

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, pretrained_embeddings, n_layers=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=True, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

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

def load_pretrained_embeddings(word_to_id, embedding_dim):
    pretrained_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    
    embedding_matrix = torch.zeros((len(word_to_id) + 1, embedding_dim))
    
    for word, idx in word_to_id.items():
        if word in pretrained_model:
            embedding_matrix[idx] = torch.tensor(pretrained_model[word])
    
    return embedding_matrix

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
    
    BATCH_SIZE = 128
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_sequence)
    valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=pad_sequence)
    
    VOCAB_SIZE = len(word_to_id) + 1
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 512
    OUTPUT_DIM = 4
    N_LAYERS = 3
    DROPOUT = 0.3
    PAD_IDX = 0
    
    pretrained_embeddings = load_pretrained_embeddings(word_to_id, EMBEDDING_DIM)
    
    model = BiRNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX, pretrained_embeddings, N_LAYERS, DROPOUT).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    N_EPOCHS = 20
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        
        scheduler.step(valid_loss)
        
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
Epoch: 01 | Epoch Time: 1m 53s
        Train Loss: 0.577 | Train Acc: 78.60%
        Valid Loss: 0.427 | Valid Acc: 85.41%
Epoch: 02 | Epoch Time: 2m 39s
        Train Loss: 0.259 | Train Acc: 91.28%
        Valid Loss: 0.328 | Valid Acc: 88.64%
Epoch: 03 | Epoch Time: 3m 26s
        Train Loss: 0.152 | Train Acc: 95.03%
        Valid Loss: 0.344 | Valid Acc: 88.33%
Epoch: 04 | Epoch Time: 2m 38s
        Train Loss: 0.095 | Train Acc: 97.05%
        Valid Loss: 0.397 | Valid Acc: 88.26%
Epoch: 05 | Epoch Time: 2m 41s
        Train Loss: 0.064 | Train Acc: 98.19%
        Valid Loss: 0.431 | Valid Acc: 85.55%
Epoch: 06 | Epoch Time: 2m 44s
        Train Loss: 0.040 | Train Acc: 98.71%
        Valid Loss: 0.578 | Valid Acc: 86.86%
Epoch: 07 | Epoch Time: 2m 39s
        Train Loss: 0.025 | Train Acc: 99.21%
        Valid Loss: 0.570 | Valid Acc: 86.55%
Epoch: 08 | Epoch Time: 2m 38s
        Train Loss: 0.019 | Train Acc: 99.40%
        Valid Loss: 0.565 | Valid Acc: 86.62%
Epoch: 09 | Epoch Time: 2m 30s
        Train Loss: 0.017 | Train Acc: 99.45%
        Valid Loss: 0.583 | Valid Acc: 86.87%
Epoch: 10 | Epoch Time: 2m 41s
        Train Loss: 0.016 | Train Acc: 99.51%
        Valid Loss: 0.581 | Valid Acc: 86.62%
Epoch: 11 | Epoch Time: 2m 42s
        Train Loss: 0.013 | Train Acc: 99.61%
        Valid Loss: 0.580 | Valid Acc: 86.86%
Epoch: 12 | Epoch Time: 2m 37s
        Train Loss: 0.012 | Train Acc: 99.64%
        Valid Loss: 0.583 | Valid Acc: 87.04%
Epoch: 13 | Epoch Time: 2m 38s
        Train Loss: 0.014 | Train Acc: 99.58%
        Valid Loss: 0.583 | Valid Acc: 86.90%
Epoch: 14 | Epoch Time: 2m 28s
        Train Loss: 0.011 | Train Acc: 99.67%
        Valid Loss: 0.589 | Valid Acc: 86.83%
Epoch: 15 | Epoch Time: 2m 20s
        Train Loss: 0.011 | Train Acc: 99.64%
        Valid Loss: 0.589 | Valid Acc: 86.83%
Epoch: 16 | Epoch Time: 2m 19s
        Train Loss: 0.013 | Train Acc: 99.58%
        Valid Loss: 0.589 | Valid Acc: 86.83%
Epoch: 17 | Epoch Time: 2m 19s
        Train Loss: 0.011 | Train Acc: 99.70%
        Valid Loss: 0.590 | Valid Acc: 86.83%
Epoch: 18 | Epoch Time: 2m 19s
        Train Loss: 0.013 | Train Acc: 99.61%
        Valid Loss: 0.590 | Valid Acc: 86.83%
Epoch: 19 | Epoch Time: 2m 17s
        Train Loss: 0.012 | Train Acc: 99.68%
        Valid Loss: 0.591 | Valid Acc: 86.83%
Epoch: 20 | Epoch Time: 2m 17s
        Train Loss: 0.014 | Train Acc: 99.56%
        Valid Loss: 0.591 | Valid Acc: 86.83%
"""