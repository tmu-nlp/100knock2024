"""
76. チェックポイント
問題75のコードを改変し，各エポックのパラメータ更新が完了するたびに，
チェックポイント（学習途中のパラメータ（重み行列など）の値や最適化アルゴリズムの内部状態）をファイルに書き出せ．
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# データの読み込み
Xtrain = torch.load('Xtrain.pt')
Ytrain = torch.load('Ytrain.pt')
Xvalid = torch.load('Xvalid.pt')
Yvalid = torch.load('Yvalid.pt')

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

# 損失と正解率を格納するリスト
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

def calculate_loss_and_accuracy(X, Y, W):
    Y_hat = F.log_softmax(torch.mm(X, W), dim=1)
    loss = F.nll_loss(Y_hat, Y)
    _, predicted = torch.max(Y_hat, 1)
    accuracy = (predicted == Y).float().mean()
    return loss.item(), accuracy.item()

# チェックポイントを保存するディレクトリ
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

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
    
    # エポックごとの訓練データでの損失と正解率を計算
    train_loss, train_accuracy = calculate_loss_and_accuracy(Xtrain, Ytrain, W)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # エポックごとの検証データでの損失と正解率を計算
    valid_loss, valid_accuracy = calculate_loss_and_accuracy(Xvalid, Yvalid, W)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)
    
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

    # チェックポイントの保存
    checkpoint = {
        'epoch': epoch + 1,
        'W': W,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'valid_loss': valid_loss,
        'valid_accuracy': valid_accuracy
    }
    torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt')

# グラフのプロット
plt.figure(figsize=(12, 5))

# 損失のプロット
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# 正解率のプロット
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 最終的な重み行列を保存
torch.save(W, 'trained_W.pt')

print("Training completed. Trained weights saved to 'trained_W.pt'")
print(f"Checkpoints saved in '{checkpoint_dir}' directory")
