import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from knock71 import SingleLayerPerceptronNetwork
from knock72 import NewsDataset, calc_loss_acc

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

    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    # モデル、損失関数、オプティマイザの初期化
    model = SingleLayerPerceptronNetwork(300, 4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # 学習
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        loss_train = 0.0
        for i, (inputs, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train = loss_train / len(dataloader_train)

        # 検証データでの評価
        loss_valid, acc_valid = calc_loss_acc(model, criterion, dataloader_valid)

        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, loss_valid: {loss_valid:.4f}, acc_valid: {acc_valid:.4f}')

    # テストデータでの最終評価
    loss_test, acc_test = calc_loss_acc(model, criterion, dataloader_test)
    print(f'\nFinal Test loss: {loss_test:.4f}, accuracy: {acc_test:.4f}')

    # 学習済みモデルの保存
    torch.save(model.state_dict(), os.path.join(current_dir, 'model.pth'))
    print("Trained model saved as 'model.pth'")

if __name__ == "__main__":
    main()

"""
epoch: 1, loss_train: 0.4588, loss_valid: 0.3946, acc_valid: 0.8666
epoch: 2, loss_train: 0.3087, loss_valid: 0.3663, acc_valid: 0.8748
epoch: 3, loss_train: 0.2802, loss_valid: 0.3402, acc_valid: 0.8861
epoch: 4, loss_train: 0.2645, loss_valid: 0.3321, acc_valid: 0.8853
epoch: 5, loss_train: 0.2530, loss_valid: 0.3254, acc_valid: 0.8936
epoch: 6, loss_train: 0.2472, loss_valid: 0.3205, acc_valid: 0.8898
epoch: 7, loss_train: 0.2419, loss_valid: 0.3199, acc_valid: 0.8951
epoch: 8, loss_train: 0.2376, loss_valid: 0.3168, acc_valid: 0.8928
epoch: 9, loss_train: 0.2342, loss_valid: 0.3167, acc_valid: 0.8981
epoch: 10, loss_train: 0.2310, loss_valid: 0.3134, acc_valid: 0.8988

Final Test loss: 0.3098, accuracy: 0.8936
Trained model saved as 'model.pth'
"""