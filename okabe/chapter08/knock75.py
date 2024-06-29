'''
75. 損失と正解率のプロットPermalink
問題73のコードを改変し，各エポックのパラメータ更新が完了するたびに，訓練データでの損失，
正解率，検証データでの損失，正解率をグラフにプロットし，学習の進捗状況を確認できるようにせよ．
'''
from tqdm import tqdm
import torch
import numpy as np
from load_vector_data import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
torch.manual_seed(seed=0)

#network
net = torch.nn.Linear(300, 4, bias=False)
loss_fn = torch.nn.CrossEntropyLoss()

#SGD
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

train_losses = []
valid_losses = []
train_accs = []
valid_accs = []

for epoch in tqdm(range(1000)):
    net.train()
    optimizer.zero_grad()
    y_pred = net(x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    net.eval()
    #loss
    train_losses.append(loss.detach().numpy())
    valid_losses.append(loss_fn(net(x_valid), y_valid).detach().numpy())
    #pred
    y_max_train, y_pred_train = torch.max(net(x_train),dim=1)
    y_max_valid, y_pred_valid = torch.max(net(x_valid),dim=1)
    #accuracy
    train_acc = accuracy_score(y_pred_train, y_train)
    valid_acc = accuracy_score(y_pred_valid, y_valid)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)

#loss plot
fig = plt.figure()
plt.plot(train_losses, label="train loss")
plt.plot(valid_losses, label="valid loss")
plt.legend()
plt.savefig("output/loss.png")

#accuracy plot
fig = plt.figure()
plt.plot(train_accs, label="train acc")
plt.plot(valid_accs, label="valid acc")
plt.legend()
plt.savefig("output/acc.png")