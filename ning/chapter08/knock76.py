from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

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

# データの読み込み
X_train = torch.load("X_train.pt")
Y_train = torch.load("Y_train.pt")
X_valid = torch.load("X_valid.pt")
Y_valid = torch.load("Y_valid.pt")

# モデルの初期化
model = NetWork(300, 4)

# 損失関数とオプティマイザの定義
loss_fn = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.SGD(params=model.parameters(), lr=0.01)

# データローダーの作成
ds = TensorDataset(X_train, Y_train)
dataloader = DataLoader(ds, batch_size=256, shuffle=True)

# 学習の実行
epoch = 1000
checkpoint_interval = 100  # チェックポイントの間隔

for ep in range(epoch):
    for X_batch, Y_batch in dataloader:
        Y_pred = model(X_batch)
        CEloss = loss_fn(Y_pred, Y_batch)
        
        optimizer.zero_grad()
        CEloss.backward()
        optimizer.step()
    
    # チェックポイントの保存
    if ep % checkpoint_interval == 0:
        checkpoint = {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': CEloss
        }
        torch.save(checkpoint, f"checkpoint_epoch_{ep:04d}.pth")

# 最終モデルの保存
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': CEloss
}, "SingleLayer_final.pth")
