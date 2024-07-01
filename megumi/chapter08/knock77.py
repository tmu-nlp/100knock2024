"""
77. ミニバッチ化
問題76のコードを改変し，B事例ごとに損失・勾配を計算し，
行列Wの値を更新せよ（ミニバッチ化）．Bの値を1,2,4,8,…と変化させながら，
1エポックの学習に要する時間を比較せよ．
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt

# データの読み込み
Xtrain = torch.load('Xtrain.pt')
Ytrain = torch.load('Ytrain.pt')

# パラメータ
d = Xtrain.shape[1]  # 特徴量の次元
L = 4  # カテゴリ数
n_epochs = 1  # エポック数（時間比較のため1エポックに設定）
learning_rate = 0.01  # 学習率

def train_epoch(batch_size):
    # 重み行列Wをランダムに初期化
    W = torch.randn(d, L, requires_grad=True)
    
    # オプティマイザの設定
    optimizer = optim.SGD([W], lr=learning_rate)
    
    start_time = time.time()
    
    for _ in range(n_epochs):
        for i in range(0, len(Xtrain), batch_size):
            # ミニバッチの取得
            batch_X = Xtrain[i:i+batch_size]
            batch_Y = Ytrain[i:i+batch_size]
            
            # 順伝播
            Y_hat = F.log_softmax(torch.mm(batch_X, W), dim=1)
            
            # 損失の計算
            loss = F.nll_loss(Y_hat, batch_Y)
            
            # 勾配の計算と更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    end_time = time.time()
    return end_time - start_time

# バッチサイズのリスト
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

# 各バッチサイズでの学習時間を計測
training_times = []
for batch_size in batch_sizes:
    time_taken = train_epoch(batch_size)
    training_times.append(time_taken)
    print(f"Batch size: {batch_size}, Time taken: {time_taken:.4f} seconds")

# 結果をグラフにプロット
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, training_times, marker='o')
plt.xscale('log', base=2)  # x軸を対数スケールに
plt.xlabel('Batch Size')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time vs Batch Size (1 Epoch)')
plt.grid(True)
plt.show()

# 結果を表形式で表示
print("\nBatch Size | Training Time (seconds)")
print("-" * 35)
for batch_size, time_taken in zip(batch_sizes, training_times):
    print(f"{batch_size:^10} | {time_taken:.4f}")
