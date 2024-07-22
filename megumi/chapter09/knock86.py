"""
86. 畳み込みニューラルネットワーク (CNN)
ID番号で表現された単語列x=(x1,x2,…,xT)がある．
ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である
（Vは単語の総数である）．
畳み込みニューラルネットワーク（CNN: Convolutional Neural Network）を用い，
単語列xからカテゴリyを予測するモデルを実装せよ．
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_filters, filter_size, num_classes):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv = nn.Conv1d(in_channels=embed_size, out_channels=num_filters, kernel_size=filter_size, padding=1)
        self.fc = nn.Linear(num_filters, num_classes)
        
    def forward(self, x):
        # 入力: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_size)
        
        # Conv1dは(batch_size, in_channels, seq_len)の入力を期待するので、埋め込みを転置
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embed_size, seq_len)
        
        # 畳み込み層
        conved = self.conv(embedded)  # (batch_size, num_filters, seq_len)
        
        # 最大値プーリング
        pooled = F.max_pool1d(conved, kernel_size=conved.shape[2]).squeeze(2)  # (batch_size, num_filters)
        
        # 全結合層
        output = self.fc(pooled)  # (batch_size, num_classes)
        
        return F.softmax(output, dim=1)

# ハイパーパラメータ
vocab_size = 10000  # 語彙サイズ
embed_size = 100    # 単語埋め込みの次元数 (dw)
num_filters = 128   # 畳み込み演算後の各時刻のベクトルの次元数 (dh)
filter_size = 3     # 畳み込みのフィルターサイズ
num_classes = 5     # カテゴリ数 (L)
seq_length = 50     # 入力文の最大長さ
batch_size = 16     # バッチサイズ

# モデルのインスタンス化
model = TextCNN(vocab_size, embed_size, num_filters, filter_size, num_classes)

# ランダムな入力データの生成
x = torch.randint(0, vocab_size, (batch_size, seq_length))

# モデルの順伝播
y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Output (probabilities):")
print(y)

