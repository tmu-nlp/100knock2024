"""
74. 正解率の計測
問題73で求めた行列を用いて学習データおよび評価データの事例を分類したとき，その正解率をそれぞれ求めよ．
"""
import torch
import torch.nn.functional as F

# データの読み込み
Xtrain = torch.load('Xtrain.pt')
Ytrain = torch.load('Ytrain.pt')
Xtest = torch.load('Xtest.pt')
Ytest = torch.load('Ytest.pt')

# 学習済みの重み行列Wを読み込む
W = torch.load('trained_W.pt')

def calculate_accuracy(X, Y, W):
    # 予測を計算
    Y_hat = F.softmax(torch.mm(X, W), dim=1)
    
    # 最も確率の高いクラスを選択
    _, predicted = torch.max(Y_hat, 1)
    
    # 正解率を計算
    correct = (predicted == Y).sum().item()
    total = Y.size(0)
    accuracy = correct / total
    
    return accuracy

# 学習データの正解率を計算
train_accuracy = calculate_accuracy(Xtrain, Ytrain, W)

# 評価データの正解率を計算
test_accuracy = calculate_accuracy(Xtest, Ytest, W)

print(f"Training Data Accuracy: {train_accuracy:.4f}")
print(f"Test Data Accuracy: {test_accuracy:.4f}")

"""
Training Data Accuracy: 0.8181
Test Data Accuracy: 0.8186
"""