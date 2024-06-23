import torch
from torch import nn
import os

class SingleLayerPerceptronNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 1.0)

    def forward(self, x):
        x = self.fc(x)
        return x

def main():
    # 現在のスクリプトのディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 学習データの特徴量行列を読み込む
    X_train = torch.load(os.path.join(current_dir, 'X_train.pt'))

    # 特徴量の次元数と分類するカテゴリの数を取得
    input_size = X_train.shape[1]  # 特徴量の次元数
    output_size = 4  # カテゴリ数（「ビジネス」「科学技術」「エンターテイメント」「健康」の4カテゴリ）

    # モデルのインスタンス化
    model = SingleLayerPerceptronNetwork(input_size, output_size)

    # 1つ目の事例x1について予測
    y_hat_1 = torch.softmax(model(X_train[:1]), dim=-1)

    # 最初の4つの事例X[1:4]について予測
    Y_hat = torch.softmax(model(X_train[:4]), dim=-1)

    print("y_hat_1 (1つ目の事例の予測結果):")
    print(y_hat_1)
    print("\nY_hat (最初の4つの事例の予測結果):")
    print(Y_hat)

    # モデルの重み行列を取得
    W = model.fc.weight.detach()

    # 重み行列を保存
    torch.save(W, os.path.join(current_dir, 'W.pt'))
    print("\n重み行列を 'W.pt' として保存しました。")

if __name__ == "__main__":
    main()

"""
y_hat_1 (1つ目の事例の予測結果):
tensor([[0.5845, 0.1099, 0.1661, 0.1394]], grad_fn=<SoftmaxBackward0>)

Y_hat (最初の4つの事例の予測結果):
tensor([[0.5845, 0.1099, 0.1661, 0.1394],
        [0.5868, 0.1187, 0.2343, 0.0602],
        [0.5593, 0.0190, 0.2055, 0.2162],
        [0.3133, 0.0288, 0.1543, 0.5036]], grad_fn=<SoftmaxBackward0>)

重み行列を 'W.pt' として保存しました。
"""