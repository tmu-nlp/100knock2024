# TensorBoardのノートブック拡張機能を読み込む
%load_ext tensorboard
%tensorboard --logdir logs

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# 訓練データおよび検証データの損失と正解率をTensorBoardに書き込む関数
def TensorboardWriter(model, X, Y, epoch, loss_fn, name):
    model.eval()
    with torch.no_grad():
        Y_pred = model(X)
        CEloss = loss_fn(Y_pred, Y)
        _, predicted = torch.max(Y_pred, 1)
        accuracy = (predicted == Y).sum().item() / len(Y)
        writer.add_scalar(f"Loss/{name}", CEloss, epoch)
        writer.add_scalar(f"Accuracy/{name}", accuracy, epoch)
    model.train()

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
X_valid = torch.load("X_valid.pt")
Y_valid = torch.load("Y_valid.pt")

# モデルの初期化
model = NetWork(300, 4)

# クロスエントロピー損失関数とSGDオプティマイザの定義
loss_fn = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.SGD(params=model.parameters(), lr=0.01)

# データローダーの作成
ds = TensorDataset(X_train, Y_train)
dataloader = DataLoader(ds, batch_size=128, shuffle=True)

# TensorBoardの設定
writer = SummaryWriter(log_dir="logs")

# 学習の実行
epoch = 100
for ep in range(epoch):
    for X_batch, Y_batch in dataloader:
        Y_pred = model(X_batch)
        CEloss = loss_fn(Y_pred, Y_batch)
        
        optimizer.zero_grad()
        CEloss.backward()
        optimizer.step()
    
    TensorboardWriter(model, X_train, Y_train, ep, loss_fn, "Train")
    TensorboardWriter(model, X_valid, Y_valid, ep, loss_fn, "Valid")

# モデルの保存
torch.save(model.state_dict(), "SingleLayer.pth")

# TensorBoardのクローズ
writer.close()
