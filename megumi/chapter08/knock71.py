"""
問題70で保存した行列を読み込み，学習データについて以下の計算を実行せよ．

ŷ 1=softmax(x1W),Ŷ =softmax(X[1:4]W)
ただし，softmax
はソフトマックス関数，X[1:4]∈ℝ4×d
は特徴ベクトルx1,x2,x3,x4
を縦に並べた行列である．

X[1:4]=⎛⎝⎜⎜⎜⎜x1x2x3x4⎞⎠⎟⎟⎟⎟
行列W∈ℝd×L
は単層ニューラルネットワークの重み行列で，ここではランダムな値で初期化すればよい（問題73以降で学習して求める）．なお，ŷ 1∈ℝL
は未学習の行列W
で事例x1
を分類したときに，各カテゴリに属する確率を表すベクトルである． 同様に，Ŷ ∈ℝn×L
は，学習データの事例x1,x2,x3,x4
について，各カテゴリに属する確率を行列として表現している．

"""

import torch
import torch.nn.functional as F

# データの読み込み
Xtrain = torch.load('Xtrain.pt')
Ytrain = torch.load('Ytrain.pt')

# パラメータ
d = Xtrain.shape[1]  # 特徴量の次元
L = 4  # カテゴリ数

# 重み行列Wをランダムに初期化
W = torch.randn(d, L, requires_grad=True)

# x1に対する予測
x1 = Xtrain[0].unsqueeze(0)  # バッチ次元を追加
y_hat_1 = F.softmax(torch.mm(x1, W), dim=1)

# X[1:4]に対する予測
X_1_4 = Xtrain[:4]
Y_hat = F.softmax(torch.mm(X_1_4, W), dim=1)

# 結果の表示
print("y_hat_1 (prediction for x1):")
print(y_hat_1)
print("\nShape of y_hat_1:", y_hat_1.shape)

print("\nY_hat (predictions for X[1:4]):")
print(Y_hat)
print("\nShape of Y_hat:", Y_hat.shape)

# カテゴリごとの確率の合計が1になることを確認
print("\nSum of probabilities for y_hat_1:", y_hat_1.sum().item())
print("Sum of probabilities for each row in Y_hat:", Y_hat.sum(dim=1))

"""
y_hat_1 (prediction for x1):
tensor([[0.7455, 0.0472, 0.1208, 0.0865]], grad_fn=<SoftmaxBackward0>)

Shape of y_hat_1: torch.Size([1, 4])

Y_hat (predictions for X[1:4]):
tensor([[0.7455, 0.0472, 0.1208, 0.0865],
        [0.2231, 0.1371, 0.6100, 0.0298],
        [0.1558, 0.1816, 0.3785, 0.2840],
        [0.3134, 0.2796, 0.0735, 0.3336]], grad_fn=<SoftmaxBackward0>)

Shape of Y_hat: torch.Size([4, 4])

Sum of probabilities for y_hat_1: 1.0
Sum of probabilities for each row in Y_hat: tensor([1.0000, 1.0000, 1.0000, 1.0000], grad_fn=<SumBackward1>)
"""