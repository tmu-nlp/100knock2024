from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import time

# 単層ニューラルネットワークの定義
class NetWork(nn.Module):
    def __init__(self, input_feature, output):
        super(NetWork, self).__init__()
        self.fc1 = nn.Linear(input_feature, output, bias=False)
        nn.init.xavier_normal_(self.fc1.weight)  # 重みをXavierの初期化
        self.fc2 = nn.Softmax(dim=1)
    
    # 線形変換
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# データの読み込み
X_train = torch.load("X_train.pt")
Y_train = torch.load("[Y_train.pt")

# 損失関数の定義
loss_fn = nn.CrossEntropyLoss(reduction="mean")

# バッチサイズのリスト
bs_list = [2**i for i in range(15)]  # 1, 2, 4, 8, ..., 16384

# バッチサイズごとの時間計測
for bs in bs_list:
    # データローダーの作成
    dataloader = DataLoader(TensorDataset(X_train, Y_train), batch_size=bs, shuffle=True)
    
    # モデルとオプティマイザの初期化
    model = NetWork(300, 4)
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)
    
    # 時間計測の開始
    start_time = time.time()
    
    # 1エポックの学習
    epoch = 1
    for ep in range(epoch):
        for X_batch, Y_batch in dataloader:
            Y_pred = model(X_batch)
            CEloss = loss_fn(Y_pred, Y_batch)
            optimizer.zero_grad()
            CEloss.backward()
            optimizer.step()
    
    # 時間計測の終了
    end_time = time.time()
    epoch_time = end_time - start_time
    
    # 結果の表示
    print(f"Batch Size: {bs}")
    print(f"Time per Epoch: {epoch_time:.6f} seconds")

