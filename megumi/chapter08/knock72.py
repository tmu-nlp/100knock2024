"""
72. 損失と勾配の計算
学習データの事例x1と事例集合x1,x2,x3,x4に対して，ク
ロスエントロピー損失と，行列Wに対する勾配を計算せよ．
なお，ある事例xiに対して損失は次式で計算される．

li=−log[事例xiがyiに分類される確率]
ただし，事例集合に対するクロスエントロピー損失は，その集合に含まれる各事例の損失の平均とする．
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

# x1に対する損失と勾配の計算
x1 = Xtrain[0].unsqueeze(0)  # バッチ次元を追加
y1 = Ytrain[0].unsqueeze(0)  # バッチ次元を追加

y_hat_1 = F.log_softmax(torch.mm(x1, W), dim=1)
loss_x1 = F.nll_loss(y_hat_1, y1)

loss_x1.backward()
gradient_x1 = W.grad.clone()

W.grad.zero_()  # 勾配をリセット

# X[1:4]に対する損失と勾配の計算
X_1_4 = Xtrain[:4]
Y_1_4 = Ytrain[:4]

Y_hat = F.log_softmax(torch.mm(X_1_4, W), dim=1)
loss_X_1_4 = F.nll_loss(Y_hat, Y_1_4)

loss_X_1_4.backward()
gradient_X_1_4 = W.grad.clone()

# 結果の表示
print("Loss for x1:", loss_x1.item())
print("Gradient for x1:")
print(gradient_x1)
print("\nShape of gradient for x1:", gradient_x1.shape)

print("\nLoss for X[1:4]:", loss_X_1_4.item())
print("Gradient for X[1:4]:")
print(gradient_X_1_4)
print("\nShape of gradient for X[1:4]:", gradient_X_1_4.shape)

"""
Loss for x1: 0.37099114060401917
Gradient for x1:
tensor([[ 0.0186, -0.0127, -0.0038, -0.0022],
        [ 0.0012, -0.0008, -0.0002, -0.0001],
        [-0.0017,  0.0011,  0.0003,  0.0002],
        ...,
        [-0.0020,  0.0014,  0.0004,  0.0002],
        [-0.0168,  0.0114,  0.0034,  0.0020],
        [ 0.0051, -0.0035, -0.0010, -0.0006]])

Shape of gradient for x1: torch.Size([300, 4])

Loss for X[1:4]: 1.1321189403533936
Gradient for X[1:4]:
tensor([[ 1.3446e-02, -8.3143e-03, -1.2351e-03, -3.8967e-03],
        [-5.1192e-03,  6.3820e-04,  3.9510e-03,  5.3005e-04],
        [-1.4779e-02,  1.0808e-05,  1.4648e-02,  1.2036e-04],
        ...,
        [-1.0366e-02, -6.5364e-03,  1.7583e-02, -6.8021e-04],
        [-2.3656e-03,  6.7796e-03, -6.9843e-03,  2.5703e-03],
        [-2.4052e-03, -1.1139e-02,  1.4736e-02, -1.1917e-03]])

Shape of gradient for X[1:4]: torch.Size([300, 4])
"""