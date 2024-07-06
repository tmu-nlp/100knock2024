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

# モデルの初期化
model = NetWork(300, 4)
model.load_state_dict(torch.load("SingleLayer.pth"))

# 学習データおよび評価データの読み込み
X_train = torch.load("X_train.pt")
Y_train = torch.load("Y_train.pt")
X_test = torch.load("X_test.pt")
Y_test = torch.load("Y_test.pt")

# 学習データに対する予測と正解率の計算
Y_train_pred = model(X_train)
train_pred_labels = torch.max(Y_train_pred, dim=1).indices
train_accuracy = train_pred_labels.eq(Y_train).sum().item() / len(Y_train)

print("訓練データの正解率:", train_accuracy)

# 評価データに対する予測と正解率の計算
Y_test_pred = model(X_test)
test_pred_labels = torch.max(Y_test_pred, dim=1).indices
test_accuracy = test_pred_labels.eq(Y_test).sum().item() / len(Y_test)

print("評価データの正解率:", test_accuracy)
