import torch
import torch.nn as nn

# 単層ニューラルネットワークの定義
class NetWork(nn.Module):
    def __init__(self, input_feature, output):
        super(NetWork, self).__init__()
        self.fc1 = nn.Linear(input_feature, output, bias=False)
        nn.init.xavier_normal_(self.fc1.weight)  # 重みをXavierの初期化
        self.fc2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 保存した特徴行列を読み込み
X_train = torch.load("X_train.pt")

# モデルの初期化
model = NetWork(300, 4)

# 最初の事例 x1 に対する予測
x1 = X_train[0].unsqueeze(0)  # x1 の形状を (1, 300) にする
y_hat_1 = model(x1)

# 最初の4事例 X[1:4] に対する予測
X_1_to_4 = X_train[:4]
Y_hat = model(X_1_to_4)

# 結果の表示
print("y_hat_1:", y_hat_1)
print("Y_hat:", Y_hat)
