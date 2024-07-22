"""
85. 双方向RNN・多層化
順方向と逆方向のRNNの両方を用いて入力テキストをエンコードし，モデルを学習せよ．
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gensim.downloader as api

# GPUが利用可能かどうかを確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, pretrained_embeddings, num_layers=1):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        
        # 初期隠れ状態を設定
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        # 双方向RNNの順伝播
        output, _ = self.rnn(embedded, h0)
        
        # 最後の時間ステップの順方向と逆方向の隠れ状態を連結
        forward = output[:, -1, :self.hidden_size]
        backward = output[:, 0, self.hidden_size:]
        combined = torch.cat((forward, backward), dim=1)
        
        # 全結合層を通して出力を得る
        out = self.fc(combined)
        return out

# 事前学習済み単語ベクトルの読み込み
print("Loading pre-trained word vectors...")
word2vec_model = api.load("word2vec-google-news-300")

# ハイパーパラメータ
vocab_size = len(word2vec_model.key_to_index)
embed_size = 300  # Google Newsの単語ベクトルは300次元
hidden_size = 128
output_size = 4
batch_size = 32
learning_rate = 0.001
num_epochs = 20
num_layers = 2  # BiRNNの層数

# 事前学習済み単語ベクトルを使用して埋め込み層を初期化
pretrained_embeddings = torch.FloatTensor(word2vec_model.vectors)

# データの準備（ダミーデータを使用）
num_samples = 1000
max_seq_length = 20

# 単語IDの代わりに単語インデックスを使用
X = torch.randint(0, vocab_size, (num_samples, max_seq_length))
y = torch.randint(0, output_size, (num_samples,))

# データを訓練セットと評価セットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# データローダーの作成
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# モデルのインスタンス化
model = BiRNN(vocab_size, embed_size, hidden_size, output_size, pretrained_embeddings, num_layers).to(device)

# 損失関数と最適化アルゴリズムの定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 学習ループ
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # 評価データでの性能評価
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    test_loss /= len(test_loader)
    test_accuracy = correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print("--------------------")

# 学習曲線のプロット
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train')
plt.plot(test_accuracies, label='Test')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
