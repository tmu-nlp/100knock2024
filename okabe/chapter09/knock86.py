'''
86. 畳み込みニューラルネットワーク (CNN)Permalink
ID番号で表現された単語列x=(x1,x2,…,xT)がある．
ただし，Tは単語列の長さ，xt∈ℝVは単語のID番号のone-hot表記である（Vは単語の総数である）．
畳み込みニューラルネットワーク（CNN: Convolutional Neural Network）を用い，
単語列xからカテゴリyを予測するモデルを実装せよ．
なお，この問題ではモデルの学習を行わず，ランダムに初期化された重み行列でyを計算するだけでよい．
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import KeyedVectors

from load_and_create_dict import *
from knock85 import df2id, list2tensor

#hyper param
max_len = 10
dw = 300
dh = 50
n_vocab = len(word2id) + 1
PAD = len(word2id) #padding_idx
epochs = 10

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.conv = torch.nn.Conv1d(dw, dh, kernel_size=3, stride=1, padding=1)#filter=3, stride=1, padding=true
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(max_len)
        self.linear = torch.nn.Linear(dh,4)
        self.softmax = torch.nn.Softmax()

    def forward(self, x, h=None):
        x = self.emb(x)
        x = x.view(x.size(0), x.size(2), x.size(1))
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        y = self.linear(x)
        y = self.softmax(y)
        return y

#model
model = CNN()

#load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#load data
X_train = df2id(train)
X_valid = df2id(valid)
X_test = df2id(test)

X_train = list2tensor(X_train,max_len)
X_valid = list2tensor(X_valid,max_len)
X_test = list2tensor(X_test,max_len)

y_train = torch.tensor(y_train, dtype=torch.int64)
y_valid = torch.tensor(y_valid, dtype=torch.int64)
y_test = torch.tensor(y_test, dtype=torch.int64)

ds = TensorDataset(X_train.to(device), y_train.to(device))

#load emb
w2v = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300_ch09.bin.gz", binary=True)
for key, val in word2id.items():
    if key in w2v.key_to_index:
        model.emb.weight[val].data = torch.tensor(w2v[key], dtype=torch.float32)
model.emb.weight = torch.nn.Parameter(model.emb.weight)

loader = DataLoader(ds, batch_size=128, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()

for xx,yy in loader:
    y_pred = model(xx)
    loss = loss_fn(y_pred, yy)
