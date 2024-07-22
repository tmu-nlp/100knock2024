"""
81. RNNによる予測
ID番号で表現された単語列x=(x1,x2,…,xT)がある．
ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である
（Vは単語の総数である）．
再帰型ニューラルネットワーク（RNN: Recurrent Neural Network）を用い，
単語列xからカテゴリyを予測するモデルとして，次式を実装せよ．
"""
import torch
import torch.nn as nn
import numpy as np

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # 単語埋め込み層
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # RNNレイヤー
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        # 出力層
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 入力をembeddingレイヤーに通す
        embedded = self.embedding(x)
        # RNNレイヤーに通す
        _, hidden = self.rnn(embedded)
        # 最後の隠れ状態を使って出力を計算
        out = self.fc(hidden.squeeze(0))
        return torch.softmax(out, dim=1)

# ハイパーパラメータ
vocab_size = 10000  # 語彙数
embed_size = 300    # 埋め込みベクトルの次元
hidden_size = 50    # RNNの隠れ状態の次元
output_size = 4     # 出力カテゴリ数

# モデルのインスタンス化
rnn = RNN(vocab_size, embed_size, hidden_size, output_size)

# テスト用の入力データ
x = torch.randint(0, vocab_size, (1, 20))  # バッチサイズ1、長さ20の単語ID列

# 予測
with torch.no_grad():
    y = rnn(x)

print("Input:", x)
print("Output (category probabilities):", y)

"""
Input: tensor([[1239, 7917, 8340, 4861, 6719, 9503, 9875, 4154,  447, 4279, 5033, 2761,
         4074,  455, 6521, 4656, 1586, 8267, 8621, 4902]])
Output (category probabilities): tensor([[0.1777, 0.2586, 0.2888, 0.2750]])
"""