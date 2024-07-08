'''
85. 双方向RNN・多層化Permalink
順方向と逆方向のRNNの両方を用いて入力テキストをエンコードし，モデルを学習せよ
さらに，双方向RNNを多層化して実験せよ．
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import KeyedVectors

from load_and_create_dict import *
from knock81 import RNN

def get_id(sentence):
    r = []
    for word in sentence:
        r.append(word2id.get(word,0))
    return r

# id list of each sentence
def df2id(df):
    ids = []
    for i in df.iloc[:,0].str.lower():
        ids.append(get_id(i.split()))
    return ids

#hyper params
max_len = 10
dw = 300 #emb dim
dh = 50 #hidden dim
n_vocab = len(word2id) + 1 #vocab
PAD = len(word2id) #padding_idx
epochs = 10

#bidirectional rnn
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(n_vocab,dw,padding_idx=PAD)
        self.rnn = torch.nn.RNN(dw,dh,batch_first=True,bidirectional=True,num_layers=3)
        self.linear = torch.nn.Linear(dh*2,4)
        self.softmax = torch.nn.Softmax()

    def forward(self, x, h=None):
        x = self.emb(x)
        y, h = self.rnn(x, h)
        y = self.linear(y[:,-1,:])
        return y

def list2tensor(data, max_len):
    new = []
    for d in data:
        if len(d) > max_len:
            d = d[:max_len]
        else:
            d += [PAD] * (max_len - len(d))
        new.append(d)
    return torch.tensor(new, dtype=torch.int64)

#acc
def accuracy(pred, label):
    pred = np.argmax(pred.cpu().data.numpy(), axis=1)
    label = label.cpu().data.numpy()
    return (pred == label).mean()

#model
model = RNN()

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

#set batch
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
        # plot
        print("valid loss: {}, valid acc: {}".format(loss.item(), accuracy(y_pred,y_valid)))


"""
output:
epoch: 0
train loss: 1.0758939981460571, train acc: 0.5596218644702359
valid loss: 1.0960147380828857, valid acc: 0.5426646706586826
epoch: 1
train loss: 1.0248862504959106, train acc: 0.6239236241108199
valid loss: 1.043318271636963, valid acc: 0.6032934131736527
epoch: 2
train loss: 0.9200301170349121, train acc: 0.6669786596780232
valid loss: 0.9475695490837097, valid acc: 0.6601796407185628
epoch: 3
train loss: 0.8836151361465454, train acc: 0.6787719955073006
valid loss: 0.8981829285621643, valid acc: 0.6796407185628742
epoch: 4
train loss: 0.7804964780807495, train acc: 0.7212654436540622
valid loss: 0.8186647295951843, valid acc: 0.7208083832335329
epoch: 5
train loss: 0.7619226574897766, train acc: 0.7287532759266192
valid loss: 0.8179284930229187, valid acc: 0.7215568862275449
epoch: 6
train loss: 0.6543794274330139, train acc: 0.7621677274429053
valid loss: 0.7395649552345276, valid acc: 0.7425149700598802
epoch: 7
train loss: 0.6893652081489563, train acc: 0.7436353425683264
valid loss: 0.8032142519950867, valid acc: 0.7110778443113772
epoch: 8
train loss: 0.6514371633529663, train acc: 0.7613253463122426
valid loss: 0.8188550472259521, valid acc: 0.7133233532934131
epoch: 9
train loss: 0.5679587125778198, train acc: 0.7972669412205167
valid loss: 0.747276246547699, valid acc: 0.7462574850299402
"""