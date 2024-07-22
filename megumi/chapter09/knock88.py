"""
88. パラメータチューニング
問題85や問題87のコードを改変し，ニューラルネットワークの形状やハイパーパラメータを調整しながら，高性能なカテゴリ分類器を構築せよ．
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import itertools

# GPUが利用可能かどうかを確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_filters, filter_sizes, num_classes, dropout):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_size, out_channels=num_filters, kernel_size=fs, padding=fs//2)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        conved = [nn.functional.relu(conv(embedded)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# ダミーデータの生成（実際のタスクでは、本物のデータセットを使用してください）
def generate_dummy_data(num_samples, seq_length, vocab_size, num_classes):
    X = torch.randint(0, vocab_size, (num_samples, seq_length))
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

# モデルの訓練と評価
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # 検証データでの評価
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        val_accuracy = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
    
    return best_val_f1

# ハイパーパラメータの探索範囲
param_grid = {
    'embed_size': [50, 100, 200],
    'num_filters': [64, 128, 256],
    'filter_sizes': [[3], [3,4], [3,4,5]],
    'dropout': [0.3, 0.5],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128]
}

# データの準備
vocab_size = 10000
num_classes = 5
seq_length = 50
num_samples = 10000
X, y = generate_dummy_data(num_samples, seq_length, vocab_size, num_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# グリッドサーチ
best_f1 = 0
best_params = {}

for params in itertools.product(*param_grid.values()):
    current_params = dict(zip(param_grid.keys(), params))
    
    # データローダーの作成
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=current_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=current_params['batch_size'])
    
    # モデルの初期化
    model = TextCNN(vocab_size, current_params['embed_size'], current_params['num_filters'], 
                    current_params['filter_sizes'], num_classes, current_params['dropout']).to(device)
    
    # 損失関数と最適化アルゴリズムの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=current_params['learning_rate'])
    
    # モデルの訓練と評価
    val_f1 = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device=device)
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_params = current_params
    
    print(f"Params: {current_params}")
    print(f"Validation F1: {val_f1:.4f}")
    print("--------------------")

print("Best Hyperparameters:")
print(best_params)
print(f"Best Validation F1: {best_f1:.4f}")

# 最適なモデルの評価
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

best_model = TextCNN(vocab_size, best_params['embed_size'], best_params['num_filters'], 
                     best_params['filter_sizes'], num_classes, best_params['dropout']).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=best_params['batch_size'], shuffle=True)
best_model_f1 = train_and_evaluate(best_model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device=device)

print(f"Best Model Test F1: {best_model_f1:.4f}")
