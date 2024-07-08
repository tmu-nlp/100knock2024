'''
79. 多層ニューラルネットワークPermalink
問題78のコードを改変し，バイアス項の導入や多層化など，
ニューラルネットワークの形状を変更しながら，高性能なカテゴリ分類器を構築せよ．
'''
from tqdm import tqdm
import torch, time
import numpy as np
from load_vector_data import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(seed=0)

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#network
net = torch.nn.Sequential(
    torch.nn.Linear(300, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 4)
)
net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
batchsize = [64]

#dataset
dataset = TensorDataset(x_train.to(device), y_train.to(device))
x_valid, y_valid = x_valid.to(device), y_valid.to(device)

times = []

train_losses = []
valid_losses = []
train_accs = []
valid_accs = []

for bs in batchsize:
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for epoch in tqdm(range(100)):
        for xx, yy in loader:
            xx, yy = xx.to(device), yy.to(device)
            optimizer.zero_grad()
            y_pred = net(xx)
            loss = loss_fn(y_pred, yy)
            loss.backward()
            optimizer.step()

        net.eval()
        #loss
        train_losses.append(loss.detach().cpu().numpy())
        valid_losses.append(loss_fn(net(x_valid), y_valid).detach().cpu().numpy())
        #pred
        y_max_train, y_pred_train = torch.max(net(x_train.to(device)),dim=1)
        y_max_valid, y_pred_valid = torch.max(net(x_valid),dim=1)
        #acc
        train_acc = accuracy_score(y_pred_train.cpu(), y_train.cpu())
        valid_acc = accuracy_score(y_pred_valid.cpu(), y_valid.cpu())
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)


#loss plot
fig = plt.figure()
plt.plot(train_losses, label="train loss")
plt.plot(valid_losses, label="valid loss")
plt.legend()
#plt.savefig("output/loss_79.png")

#acc plot
fig = plt.figure()
plt.plot(train_accs, label="train acc")
plt.plot(valid_accs, label="valid acc")
plt.legend()
#plt.savefig("output/acc_79.png")