import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from knock71 import SingleLayerPerceptronNetwork
from knock72 import NewsDataset, calc_loss_acc

def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs):
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        valid_loss, valid_acc = calc_loss_acc(model, criterion, valid_loader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')

    return train_losses, train_accs, valid_losses, valid_accs

def plot_learning_curves(train_losses, train_accs, valid_losses, valid_accs):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(valid_accs, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # データの読み込み
    X_train = torch.load(os.path.join(current_dir, "X_train.pt"))
    X_valid = torch.load(os.path.join(current_dir, "X_valid.pt"))
    X_test = torch.load(os.path.join(current_dir, "X_test.pt"))
    y_train = torch.load(os.path.join(current_dir, "y_train.pt"))
    y_valid = torch.load(os.path.join(current_dir, "y_valid.pt"))
    y_test = torch.load(os.path.join(current_dir, "y_test.pt"))

    # データセットとデータローダーの作成
    dataset_train = NewsDataset(X_train, y_train)
    dataset_valid = NewsDataset(X_valid, y_valid)
    dataset_test = NewsDataset(X_test, y_test)

    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    # モデル、損失関数、オプティマイザの初期化
    model = SingleLayerPerceptronNetwork(300, 4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # 学習
    num_epochs = 30
    train_losses, train_accs, valid_losses, valid_accs = train_model(
        model, criterion, optimizer, dataloader_train, dataloader_valid, num_epochs)

    # 学習曲線のプロット
    plot_learning_curves(train_losses, train_accs, valid_losses, valid_accs)

    # テストデータでの最終評価
    loss_test, acc_test = calc_loss_acc(model, criterion, dataloader_test)
    print(f'\nFinal Test loss: {loss_test:.4f}, accuracy: {acc_test:.4f}')

    # 学習済みモデルの保存
    torch.save(model.state_dict(), os.path.join(current_dir, 'model.pth'))
    print("Trained model saved as 'model.pth'")

if __name__ == "__main__":
    main()