import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from knock81 import NewsDataset, load_data, create_word_to_id, pad_sequence
import os

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
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.permute(0, 2, 1)
        # embedded = [batch size, emb dim, sent len]
        
        conved = F.relu(self.conv(embedded))
        # conved = [batch size, n_filters, sent len]
        
        pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
        # pooled = [batch size, n_filters]
        
        return self.fc(pooled)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    base_path = os.path.join('..', 'chapter06')
    train_data = load_data(os.path.join(base_path, 'train.txt'))
    
    word_to_id = create_word_to_id(train_data)
    category_to_id = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    
    train_dataset = NewsDataset(train_data['TITLE'], train_data['CATEGORY'].map(category_to_id), word_to_id)
    
    BATCH_SIZE = 64
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_sequence)
    
    VOCAB_SIZE = len(word_to_id) + 1
    EMBEDDING_DIM = 300
    N_FILTERS = 100
    FILTER_SIZE = 3
    OUTPUT_DIM = 4
    PAD_IDX = 0
    
    model = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, N_FILTERS, FILTER_SIZE, OUTPUT_DIM, PAD_IDX).to(device)
    
    # ランダムに初期化された重み行列でyを計算
    for batch in train_iterator:
        texts, labels, _ = [x.to(device) for x in batch]
        
        predictions = model(texts)
        y = F.softmax(predictions, dim=1)
        
        print("Sample input shape:", texts.shape)
        print("Sample output shape:", y.shape)
        print("Sample output (first 5 examples):")
        print(y[:5])
        
        break  # 1バッチだけ処理して終了

if __name__ == "__main__":
    main()