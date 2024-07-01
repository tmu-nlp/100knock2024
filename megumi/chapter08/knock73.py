"""
73. 確率的勾配降下法による学習
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，行列Wを学習せよ．
なお，学習は適当な基準で終了させればよい（例えば「100エポックで終了」など）．
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

# データの読み込み
Xtrain = torch.load('Xtrain.pt')
Ytrain = torch.load('Ytrain.pt')

# パラメータ
d = Xtrain.shape[1]  # 特徴量の次元
L = 4  # カテゴリ数
n_epochs = 100  # エポック数
batch_size = 32  # バッチサイズ
learning_rate = 0.01  # 学習率

# 重み行列Wをランダムに初期化
W = torch.randn(d, L, requires_grad=True)

# オプティマイザの設定
optimizer = optim.SGD([W], lr=learning_rate)

# 学習ループ
for epoch in range(n_epochs):
    total_loss = 0
    for i in range(0, len(Xtrain), batch_size):
        # ミニバッチの取得
        batch_X = Xtrain[i:i+batch_size]
        batch_Y = Ytrain[i:i+batch_size]
        
        # 順伝播
        Y_hat = F.log_softmax(torch.mm(batch_X, W), dim=1)
        
        # 損失の計算
        loss = F.nll_loss(Y_hat, batch_Y)
        total_loss += loss.item()
        
        # 勾配の計算と更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # エポックごとの平均損失を表示
    avg_loss = total_loss / (len(Xtrain) / batch_size)
    print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")

# 学習後の重み行列を保存
torch.save(W, 'trained_W.pt')

print("Training completed. Trained weights saved to 'trained_W.pt'")
