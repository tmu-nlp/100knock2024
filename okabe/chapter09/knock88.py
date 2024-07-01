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

"""
output:
epoch: 0
train loss: 1.144091010093689, train acc: 0.49298015724447775
valid loss: 1.1529046297073364, valid acc: 0.48727544910179643
epoch: 1
train loss: 1.111140251159668, train acc: 0.5292025458629727
valid loss: 1.1263147592544556, valid acc: 0.5097305389221557
epoch: 2
train loss: 1.0401456356048584, train acc: 0.5887308124298015
valid loss: 1.0628989934921265, valid acc: 0.5763473053892215
epoch: 3
train loss: 0.9125668406486511, train acc: 0.6720329464619993
valid loss: 0.9180110692977905, valid acc: 0.6714071856287425
epoch: 4
train loss: 0.8932933807373047, train acc: 0.6744664919505803
valid loss: 0.9155377745628357, valid acc: 0.6736526946107785
epoch: 5
train loss: 0.8822873830795288, train acc: 0.6762448521153126
valid loss: 0.8971542716026306, valid acc: 0.6781437125748503
epoch: 6
train loss: 0.8523778915405273, train acc: 0.6837326843878697
valid loss: 0.8948400020599365, valid acc: 0.6736526946107785
epoch: 7
train loss: 0.8020069599151611, train acc: 0.711437663796331
valid loss: 0.8467804789543152, valid acc: 0.6998502994011976
epoch: 8
train loss: 0.8492867946624756, train acc: 0.6741856982403595
valid loss: 0.8995225429534912, valid acc: 0.6706586826347305
epoch: 9
train loss: 0.8038095831871033, train acc: 0.6917821040808686
valid loss: 0.8635315299034119, valid acc: 0.6669161676646707
epoch: 10
train loss: 0.6772409677505493, train acc: 0.7531823287158368
valid loss: 0.7634317874908447, valid acc: 0.7312874251497006
epoch: 11
train loss: 0.6606354117393494, train acc: 0.7569262448521153
valid loss: 0.755976140499115, valid acc: 0.7260479041916168
epoch: 12
train loss: 0.6763549447059631, train acc: 0.7566454511418944
valid loss: 0.7842397093772888, valid acc: 0.7065868263473054
epoch: 13
train loss: 0.6230843663215637, train acc: 0.7707787345563459
valid loss: 0.7529125809669495, valid acc: 0.7357784431137725
epoch: 14
train loss: 0.6001920104026794, train acc: 0.7814488955447398
valid loss: 0.7602198123931885, valid acc: 0.7455089820359282
epoch: 15
train loss: 0.5522139668464661, train acc: 0.8004492699363535
valid loss: 0.7287760972976685, valid acc: 0.7320359281437125
epoch: 16
train loss: 0.576865017414093, train acc: 0.7836952452265069
valid loss: 0.7774217128753662, valid acc: 0.7275449101796407
epoch: 17
train loss: 0.5954979062080383, train acc: 0.7816360913515538
valid loss: 0.79807049036026, valid acc: 0.7432634730538922
epoch: 18
train loss: 0.530119776725769, train acc: 0.80147884687383
valid loss: 0.76053786277771, valid acc: 0.7342814371257484
epoch: 19
train loss: 0.4862574636936188, train acc: 0.8202920254586298
valid loss: 0.7392767667770386, valid acc: 0.7380239520958084
epoch: 20
train loss: 0.49515777826309204, train acc: 0.8153313365780607
valid loss: 0.7700255513191223, valid acc: 0.7260479041916168
epoch: 21
train loss: 0.42582380771636963, train acc: 0.8508985398727068
valid loss: 0.7205601930618286, valid acc: 0.7380239520958084
epoch: 22
train loss: 0.6918822526931763, train acc: 0.742324971920629
valid loss: 0.9771093726158142, valid acc: 0.6901197604790419
epoch: 23
train loss: 0.3913634419441223, train acc: 0.852864095844253
valid loss: 0.760977029800415, valid acc: 0.7335329341317365
epoch: 24
train loss: 0.38211914896965027, train acc: 0.8625982777985773
valid loss: 0.7530995607376099, valid acc: 0.7410179640718563
epoch: 25
train loss: 0.4232211410999298, train acc: 0.8471546237364284
valid loss: 0.7979640364646912, valid acc: 0.7230538922155688
epoch: 26
train loss: 0.41911858320236206, train acc: 0.8348932983901161
valid loss: 0.812635064125061, valid acc: 0.7125748502994012
epoch: 27
train loss: 0.3129066228866577, train acc: 0.8830026207412954
valid loss: 0.7575474381446838, valid acc: 0.749251497005988
epoch: 28
train loss: 0.36172232031822205, train acc: 0.8681205540995882
valid loss: 0.8044009804725647, valid acc: 0.7305389221556886
epoch: 29
train loss: 0.33500468730926514, train acc: 0.8715836765256458
valid loss: 0.8086695671081543, valid acc: 0.7372754491017964
epoch: 30
train loss: 0.4669731557369232, train acc: 0.8200112317484088
valid loss: 0.9749253392219543, valid acc: 0.7058383233532934
epoch: 31
train loss: 0.3068205416202545, train acc: 0.8897416697865967
valid loss: 0.8652718663215637, valid acc: 0.7230538922155688
epoch: 32
train loss: 0.3980758488178253, train acc: 0.8514601272931487
valid loss: 0.9022820591926575, valid acc: 0.7005988023952096
epoch: 33
train loss: 1.1096302270889282, train acc: 0.6551853238487458
valid loss: 1.5938284397125244, valid acc: 0.594311377245509
epoch: 34
train loss: 0.24317266047000885, train acc: 0.9165106701609884
valid loss: 0.8341537117958069, valid acc: 0.7485029940119761
epoch: 35
train loss: 0.27174532413482666, train acc: 0.8933919880194684
valid loss: 0.8990411162376404, valid acc: 0.750748502994012
epoch: 36
train loss: 0.429683655500412, train acc: 0.8320853612879071
valid loss: 1.067908525466919, valid acc: 0.688622754491018
epoch: 37
train loss: 0.5168784260749817, train acc: 0.809247472856608
valid loss: 1.035555124282837, valid acc: 0.719311377245509
epoch: 38
train loss: 0.18439412117004395, train acc: 0.9394421564956945
valid loss: 0.8763262629508972, valid acc: 0.7552395209580839
epoch: 39
train loss: 0.1741844117641449, train acc: 0.9412205166604268
valid loss: 0.9140494465827942, valid acc: 0.7537425149700598
"""
