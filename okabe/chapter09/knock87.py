'''
87. 確率的勾配降下法によるCNNの学習Permalink
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，
問題86で構築したモデルを学習せよ．訓練データ上の損失と正解率，
評価データ上の損失と正解率を表示しながらモデルを学習し，適当な基準（例えば10エポックなど）で終了させよ．
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import KeyedVectors

from load_and_create_dict import *
from knock85 import df2id, list2tensor, accuracy
from knock86 import CNN

#hyper param
max_len = 10
dw = 300
dh = 50
n_vocab = len(word2id) + 1
PAD = len(word2id) #padding_idx
epochs = 10

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
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

#train model
for epoch in range(epochs):
    for xx, yy in loader:
        y_pred = model(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_pred = model(X_train.to(device))
        loss = loss_fn(y_pred, y_train.to(device))
        print("epoch: {}".format(epoch))
        print("train loss: {}, train acc: {}".format(loss.item(), accuracy(y_pred,y_train)))
        y_pred = model(X_valid.to(device))
        loss = loss_fn(y_pred, y_valid.to(device))
        print("valid loss: {}, valid acc: {}".format(loss.item(), accuracy(y_pred,y_valid)))

"""
output:
epoch: 0
train loss: 1.1102229356765747, train acc: 0.6379633096218644
valid loss: 1.1133068799972534, valid acc: 0.6444610778443114
epoch: 1
train loss: 1.0737571716308594, train acc: 0.6781168101834519
valid loss: 1.087797999382019, valid acc: 0.6549401197604791
epoch: 2
train loss: 1.0528749227523804, train acc: 0.6986147510295769
valid loss: 1.0762286186218262, valid acc: 0.6699101796407185
epoch: 3
train loss: 1.0307334661483765, train acc: 0.7273493073755148
valid loss: 1.0637717247009277, valid acc: 0.6788922155688623
epoch: 4
train loss: 1.0172148942947388, train acc: 0.7386746536877574
valid loss: 1.0637377500534058, valid acc: 0.6766467065868264
epoch: 5
train loss: 1.0033009052276611, train acc: 0.7538375140396855
valid loss: 1.0565468072891235, valid acc: 0.6841317365269461
epoch: 6
train loss: 0.9991310238838196, train acc: 0.7558030700112317
valid loss: 1.0607140064239502, valid acc: 0.6751497005988024
epoch: 7
train loss: 0.9870043992996216, train acc: 0.768064395357544
valid loss: 1.0538017749786377, valid acc: 0.6893712574850299
epoch: 8
train loss: 0.9808890223503113, train acc: 0.7722763010108573
valid loss: 1.0559066534042358, valid acc: 0.6826347305389222
epoch: 9
train loss: 0.9802418947219849, train acc: 0.7726506926244852
valid loss: 1.059266209602356, valid acc: 0.6766467065868264
"""