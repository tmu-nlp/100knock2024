'''
88. パラメータチューニングPermalink
問題85や問題87のコードを改変し，ニューラルネットワークの形状やハイパーパラメータを調整しながら，
高性能なカテゴリ分類器を構築せよ．
'''
import torch
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import KeyedVectors

from load_and_create_dict import *
from knock85 import df2id, list2tensor, accuracy

#hyper param
max_len = 10
dw = 300
dh = 50
n_vocab = len(word2id) + 1
PAD = len(word2id) #padding_idx
epochs = 40 #epoch10->40

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

#model
model = RNN()

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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

loader = DataLoader(ds, batch_size=256, shuffle=True)
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