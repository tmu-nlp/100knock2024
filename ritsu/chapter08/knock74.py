import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from knock71 import SingleLayerPerceptronNetwork
from knock72 import NewsDataset, calc_acc

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # データの読み込み
    X_train = torch.load(os.path.join(current_dir, "X_train.pt"))
    X_test = torch.load(os.path.join(current_dir, "X_test.pt"))
    y_train = torch.load(os.path.join(current_dir, "y_train.pt"))
    y_test = torch.load(os.path.join(current_dir, "y_test.pt"))

    # データセットとデータローダーの作成
    dataset_train = NewsDataset(X_train, y_train)
    dataset_test = NewsDataset(X_test, y_test)

    dataloader_train = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    # モデルの初期化と学習済みの重みの読み込み
    model = SingleLayerPerceptronNetwork(300, 4)
    model.load_state_dict(torch.load(os.path.join(current_dir, 'model.pth')))
    model.eval()  # 評価モードに設定

    # 学習データの正解率を計算
    acc_train = calc_acc(model, dataloader_train)

    # 評価データの正解率を計算
    acc_test = calc_acc(model, dataloader_test)

    print(f"学習データの正解率: {acc_train:.4f}")
    print(f"評価データの正解率: {acc_test:.4f}")

if __name__ == "__main__":
    main()

"""
学習データの正解率: 0.9243
評価データの正解率: 0.8936
"""