"""
79. 多層ニューラルネットワーク
問題78のコードを改変し，バイアス項の導入や多層化など，ニューラルネットワークの形状を変更しながら，高性能なカテゴリ分類器を構築せよ
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

# GPUが利用可能かどうかを確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# データの読み込みとGPUへの移動
Xtrain = torch.load('Xtrain.pt').to(device)
Ytrain = torch.load('Ytrain.pt').to(device)
Xvalid = torch.load('Xvalid.pt').to(device)
Yvalid = torch.load('Yvalid.pt').to(device)

# パラメータ
input_dim = Xtrain.shape[1]  # 入力の次元
hidden_dim = 100  # 隠れ層のユニット数
output_dim = 4  # カテゴリ数
n_epochs = 50  # エポック数
batch_size = 64  # バッチサイズ
learning_rate = 0.001  # 学習率

# 多層ニューラルネットワークの定義
class MultiLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# モデルの初期化
model = MultiLayerNet(input_dim, hidden_dim, output_dim).to(device)

# 損失関数とオプティマイザの設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 学習ループ
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i in range(0, len(Xtrain), batch_size):
        batch_X = Xtrain[i:i+batch_size]
        batch_Y = Ytrain[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_Y.size(0)
        correct += (predicted == batch_Y).sum().item()

    train_loss = total_loss / (len(Xtrain) / batch_size)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 検証データでの評価
    model.eval()
    with torch.no_grad():
        valid_outputs = model(Xvalid)
        valid_loss = criterion(valid_outputs, Yvalid).item()
        _, predicted = torch.max(valid_outputs.data, 1)
        valid_accuracy = (predicted == Yvalid).float().mean().item()

    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

    print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
          f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}')

# 学習曲線のプロット
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(valid_accuracies, label='Valid Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# テストデータでの評価
# テストデータでの評価
Xtest = torch.load('Xtest.pt').to(device)
Ytest = torch.load('Ytest.pt').to(device)

model.eval()
with torch.no_grad():
    test_outputs = model(Xtest)
    _, predicted = torch.max(test_outputs.data, 1)
    test_accuracy = (predicted == Ytest).float().mean().item()

print(f'Test Accuracy: {test_accuracy:.4f}')
