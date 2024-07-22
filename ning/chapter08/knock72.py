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

# 保存した特徴行列とラベルベクトルを読み込み
X_train = torch.load("X_train.pt")
Y_train = torch.load("Y_train.pt")

# モデルの初期化
model = NetWork(300, 4)

# クロスエントロピー損失関数の定義
loss_fn = nn.CrossEntropyLoss(reduction="mean")

# 最初の事例 x1 に対する損失と勾配の計算
x1 = X_train[0].unsqueeze(0)  # x1 の形状を (1, 300) にする
y1 = Y_train[0].unsqueeze(0)  # y1 の形状を (1,) にする

model.zero_grad()
y_hat_1 = model(x1)
loss_x1 = loss_fn(y_hat_1, y1)
loss_x1.backward()

print("事例 x1 に対するクロスエントロピー損失:", loss_x1.item())
print("事例 x1 に対する勾配:", model.fc1.weight.grad)

# 最初の4事例 X[1:4] に対する損失と勾配の計算
X_1_to_4 = X_train[:4]
Y_1_to_4 = Y_train[:4]

model.zero_grad()
Y_hat_1_to_4 = model(X_1_to_4)
loss_X_1_to_4 = loss_fn(Y_hat_1_to_4, Y_1_to_4)
loss_X_1_to_4.backward()

print("事例集合 x1, x2, x3, x4 に対するクロスエントロピー損失:", loss_X_1_to_4.item())
print("事例集合 x1, x2, x3, x4 に対する勾配:", model.fc1.weight.grad)
