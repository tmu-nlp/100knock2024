import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# 多層パーセプトロンネットワークの定義
class MultiLayerPerceptronNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # 入力層から最初の隠れ層への線形変換
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        # 隠れ層間の線形変換
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        # 最後の隠れ層から出力層への線形変換
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        # 各隠れ層に対応するバッチ正規化層
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in hidden_sizes])

    def forward(self, x):
        # 順伝播の定義
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.bn_layers)):
            # 線形変換 -> バッチ正規化 -> ReLU活性化
            x = F.relu(bn(layer(x)))
        # 最終層（出力層）の線形変換
        x = self.layers[-1](x)
        return x

# ニュースデータセットのカスタム定義
class NewsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]

# モデルの精度を計算する関数
def calc_acc(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return correct / total

# モデルの損失と精度を計算する関数
def calc_loss_acc(model, criterion, loader, device):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return loss / len(loader), correct / total

# モデルの学習を行う関数
def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device):
    model.to(device)
    
    # データローダーの設定
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
    
    log_train = []
    log_valid = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        model.train()
        for inputs, labels in dataloader_train:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        
        # 訓練データと検証データの損失と精度を計算
        loss_train, acc_train = calc_loss_acc(model, criterion, dataloader_train, device)
        loss_valid, acc_valid = calc_loss_acc(model, criterion, dataloader_valid, device)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])
        
        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, '
              f'loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, '
              f'train_time: {(end_time - start_time):.4f}sec')
    
    return model, log_train, log_valid

def main():
    # GPUが利用可能な場合はGPUを、そうでない場合はCPUを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データの読み込み
    X_train = torch.load("X_train.pt")
    X_valid = torch.load("X_valid.pt")
    X_test = torch.load("X_test.pt")
    y_train = torch.load("y_train.pt")
    y_valid = torch.load("y_valid.pt")
    y_test = torch.load("y_test.pt")

    # データセットの作成
    dataset_train = NewsDataset(X_train, y_train)
    dataset_valid = NewsDataset(X_valid, y_valid)
    dataset_test = NewsDataset(X_test, y_test)

    # モデルのパラメータ設定
    input_size = 300
    hidden_sizes = [200]  # 中間層のユニット数
    # hidden_sizes = [1000, 500, 200]
    # hidden_sizes = [200, 100] 
    output_size = 4
    model = MultiLayerPerceptronNetwork(input_size, hidden_sizes, output_size)
    criterion = nn.CrossEntropyLoss()
    
    # オプティマイザの選択（コメントアウトされた行は他のオプションを示しています）
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    num_epochs = 50
    batch_size = 64

    # モデルの学習
    model, log_train, log_valid = train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device)

    # 学習曲線のプロット
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.array(log_train)[:, 0], label='train')
    ax[0].plot(np.array(log_valid)[:, 0], label='valid')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].legend()
    ax[1].plot(np.array(log_train)[:, 1], label='train')
    ax[1].plot(np.array(log_valid)[:, 1], label='valid')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()
    plt.savefig("79_improved.png")

    # 最終的な精度の計算
    dataloader_train = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
    acc_train = calc_acc(model, dataloader_train, device)
    acc_test = calc_acc(model, dataloader_test, device)
    print(f"train_acc : {acc_train}")
    print(f"test_acc : {acc_test}")

if __name__ == "__main__":
    main()

"""
SGD lr=1e-1 hidden_sizes=[200] epoch=50
train_acc : 0.9990629685157422
test_acc : 0.9227886056971514

SGD lr=1e-2 hidden_sizes=[1000, 500, 200] epoch=50
train_acc : 0.9989692653673163
test_acc : 0.9212893553223388
"""