from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class MultiLayerNetwork(nn.Module):
    def __init__(self, input_feature, output):
        super(MultiLayerNetwork, self).__init__()
        self.fc1 = nn.Linear(input_feature, 512, bias=True)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc3 = nn.Linear(256, 128, bias=True)
        self.fc4 = nn.Linear(128, output, bias=True)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        return x

# データの読み込み
X_train = torch.load("[PATH]/X_train.pt").to("cuda:0")
Y_train = torch.load("[PATH]/Y_train.pt").to("cuda:0")
X_test = torch.load("[PATH]/X_test.pt").to("cuda:0")
Y_test = torch.load("[PATH]/Y_test.pt").to("cuda:0")

# 損失関数の定義
loss_fn = nn.CrossEntropyLoss(reduction="mean")

# モデルの初期化
model = MultiLayerNetwork(300, 4).to("cuda:0")

# データローダーの作成
ds = TensorDataset(X_train, Y_train)
dataloader = DataLoader(ds, batch_size=256, shuffle=True)

# オプティマイザの定義
optimizer = optim.SGD(params=model.parameters(), lr=0.001)

# TensorBoardの設定
writer = SummaryWriter(log_dir="logs")

# 学習の実行
epoch = 1000
for ep in range(epoch):
    model.train()
    running_loss = 0.0
    for X_batch, Y_batch in dataloader:
        Y_pred = model(X_batch)
        CEloss = loss_fn(Y_pred, Y_batch)
        optimizer.zero_grad()
        CEloss.backward()
        optimizer.step()
        running_loss += CEloss.item()
    
    # エポックごとの損失と正解率の計算
    with torch.no_grad():
        model.eval()
        Y_train_pred = model(X_train)
        train_loss = loss_fn(Y_train_pred, Y_train).item()
        train_acc = (torch.max(Y_train_pred, dim=1).indices == Y_train).sum().item() / len(Y_train)
        
        Y_test_pred = model(X_test)
        test_loss = loss_fn(Y_test_pred, Y_test).item()
        test_acc = (torch.max(Y_test_pred, dim=1).indices == Y_test).sum().item() / len(Y_test)
    
    writer.add_scalar('Loss/train', train_loss, ep)
    writer.add_scalar('Loss/test', test_loss, ep)
    writer.add_scalar('Accuracy/train', train_acc, ep)
    writer.add_scalar('Accuracy/test', test_acc, ep)
    
    if ep % 100 == 0:
        print(f"Epoch {ep+1}/{epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# TensorBoardのクローズ
writer.close()

# 最終モデルの保存
torch.save(model.state_dict(), "[PATH]/MultiLayerModel_final.pth")

# CPU上での精度計算
model.cpu()
X_train = X_train.cpu()
Y_train = Y_train.cpu()
X_test = X_test.cpu()
Y_test = Y_test.cpu()

with torch.no_grad():
    Y_train_pred = model(X_train)
    train_acc = (torch.max(Y_train_pred, dim=1).indices == Y_train).sum().item() / len(Y_train)
    print("学習データの正解率:", train_acc)

    Y_test_pred = model(X_test)
    test_acc = (torch.max(Y_test_pred, dim=1).indices == Y_test).sum().item() / len(Y_test)
    print("評価データの正解率:", test_acc)
