import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
from knock71 import SingleLayerPerceptronNetwork
from knock72 import NewsDataset, calc_loss_acc

def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        valid_loss, valid_acc = calc_loss_acc(model, criterion, valid_loader)

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, '
              f'Time: {epoch_time:.2f}s')

    return epoch_time

def plot_batch_times(batch_sizes, times):
    plt.figure(figsize=(10, 5))
    plt.plot(batch_sizes, times, marker='o')
    plt.title('Training Time per Epoch vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.xscale('log', base=2)
    plt.grid(True)
    plt.savefig('batch_times.png')
    plt.show()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # データの読み込み
    X_train = torch.load(os.path.join(current_dir, "X_train.pt"))
    X_valid = torch.load(os.path.join(current_dir, "X_valid.pt"))
    y_train = torch.load(os.path.join(current_dir, "y_train.pt"))
    y_valid = torch.load(os.path.join(current_dir, "y_valid.pt"))

    # データセットの作成
    dataset_train = NewsDataset(X_train, y_train)
    dataset_valid = NewsDataset(X_valid, y_valid)

    # バッチサイズのリスト
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    times = []

    for batch_size in batch_sizes:
        print(f"\nTraining with batch size: {batch_size}")
        
        # データローダーの作成
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

        # モデル、損失関数、オプティマイザの初期化
        model = SingleLayerPerceptronNetwork(300, 4)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

        # 学習と時間計測
        epoch_time = train_model(model, criterion, optimizer, dataloader_train, dataloader_valid, num_epochs=1)
        times.append(epoch_time)

    # 結果のプロット
    plot_batch_times(batch_sizes, times)

if __name__ == "__main__":
    main()