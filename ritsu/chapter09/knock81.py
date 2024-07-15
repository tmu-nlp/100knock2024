import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return self.fc(hidden.squeeze(0))

class NewsDataset(Dataset):
    def __init__(self, texts, labels, word_to_id):
        self.texts = texts
        self.labels = labels
        self.word_to_id = word_to_id
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = text.split()
        ids = [self.word_to_id.get(token, 0) for token in tokens]
        
        return torch.tensor(ids), label, len(ids)

def load_data(file_path):
    return pd.read_csv(file_path, sep='\t', names=['CATEGORY', 'TITLE'])

def create_word_to_id(train_data):
    word_count = defaultdict(int)
    for title in train_data['TITLE']:
        for word in title.split():
            word_count[word] += 1
    
    word_to_id = {word: i+1 for i, (word, count) in enumerate(sorted(word_count.items(), key=lambda x: x[1], reverse=True)) if count >= 2}
    return word_to_id

def pad_sequence(batch):
    texts, labels, lengths = zip(*batch)
    padded_texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return padded_texts, torch.tensor(labels), torch.tensor(lengths)

def main():
    base_path = os.path.join('..', 'chapter06')
    train_data = load_data(os.path.join(base_path, 'train.txt'))
    valid_data = load_data(os.path.join(base_path, 'valid.txt'))
    test_data = load_data(os.path.join(base_path, 'test.txt'))

    word_to_id = create_word_to_id(train_data)
    category_to_id = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    train_dataset = NewsDataset(train_data['TITLE'], train_data['CATEGORY'].map(category_to_id), word_to_id)
    valid_dataset = NewsDataset(valid_data['TITLE'], valid_data['CATEGORY'].map(category_to_id), word_to_id)
    test_dataset = NewsDataset(test_data['TITLE'], test_data['CATEGORY'].map(category_to_id), word_to_id)

    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_sequence)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=pad_sequence)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=pad_sequence)

    VOCAB_SIZE = len(word_to_id) + 1  # +1 for padding
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 50
    OUTPUT_DIM = 4
    PAD_IDX = 0

    model = RNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX)

    # モデルのテスト
    for batch in train_loader:
        texts, labels, lengths = batch
        output = model(texts, lengths)
        print("Output shape:", output.shape)
        print("Sample output:", torch.softmax(output[0], dim=0))
        break

if __name__ == "__main__":
    main()