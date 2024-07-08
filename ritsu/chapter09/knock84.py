import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from knock81 import NewsDataset, load_data, create_word_to_id, pad_sequence
import os
import time
import gensim

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, pretrained_embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return self.fc(hidden.squeeze(0))

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
    # Google Newsの事前学習済み単語ベクトルをロード
    pretrained_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    
    # 単語埋め込み行列を初期化
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
    
    BATCH_SIZE = 64
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_sequence)
    valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=pad_sequence)
    
    VOCAB_SIZE = len(word_to_id) + 1
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 50
    OUTPUT_DIM = 4
    PAD_IDX = 0
    
    # 事前学習済み単語ベクトルをロード
    pretrained_embeddings = load_pretrained_embeddings(word_to_id, EMBEDDING_DIM)
    
    model = RNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX, pretrained_embeddings).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
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
Epoch: 01 | Epoch Time: 0m 2s
        Train Loss: 1.229 | Train Acc: 46.54%
        Valid Loss: 1.150 | Valid Acc: 52.68%
Epoch: 02 | Epoch Time: 0m 2s
        Train Loss: 1.111 | Train Acc: 58.88%
        Valid Loss: 1.097 | Valid Acc: 59.25%
Epoch: 03 | Epoch Time: 0m 2s
        Train Loss: 1.050 | Train Acc: 64.91%
        Valid Loss: 1.020 | Valid Acc: 65.36%
Epoch: 04 | Epoch Time: 0m 1s
        Train Loss: 0.817 | Train Acc: 73.85%
        Valid Loss: 0.706 | Valid Acc: 76.04%
Epoch: 05 | Epoch Time: 0m 1s
        Train Loss: 0.647 | Train Acc: 77.47%
        Valid Loss: 0.703 | Valid Acc: 74.74%
Epoch: 06 | Epoch Time: 0m 1s
        Train Loss: 0.611 | Train Acc: 77.98%
        Valid Loss: 0.655 | Valid Acc: 77.15%
Epoch: 07 | Epoch Time: 0m 1s
        Train Loss: 0.589 | Train Acc: 78.26%
        Valid Loss: 0.614 | Valid Acc: 78.29%
Epoch: 08 | Epoch Time: 0m 1s
        Train Loss: 0.572 | Train Acc: 78.87%
        Valid Loss: 0.599 | Valid Acc: 78.74%
Epoch: 09 | Epoch Time: 0m 1s
        Train Loss: 0.553 | Train Acc: 79.28%
        Valid Loss: 0.618 | Valid Acc: 77.84%
Epoch: 10 | Epoch Time: 0m 1s
        Train Loss: 0.537 | Train Acc: 79.68%
        Valid Loss: 0.576 | Valid Acc: 78.36%
"""