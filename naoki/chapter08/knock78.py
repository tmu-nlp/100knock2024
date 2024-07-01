from tqdm import tqdm
import torch, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

class Network(nn.Module):
    def __init__(self,input_feature,output):
        super().__init__()
        self.fc1 = nn.Linear(input_feature, output)
        nn.init.normal_(self.fc1.weight, mean=0, std=1)
        self.fc2 = nn.Softmax(dim=1)
    def forward(self,x):
        y = self.fc1(x)
        z = self.fc2(y)
        return z

X_train = torch.load('X_train.pt')
y_train = torch.load('y_train.pt')
model = Network(300,4)
model = model.to('cuda:0')  # ココだけ変更
loss = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.05)

batchsize = [2**i for i in range(5)]

for bs in batchsize:
    #TensorDatasetは２つのテンソルをデータセットにまとめる
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True)
    loss_train = []
    acc_train = []
    loss_valid = []
    acc_valid = []

    t1 = time.time()

    for epoc in range(1):
        start = time.time()
        for X,Y in loader:
            Y_pred = model(X)
            CEloss = loss(Y_pred, Y)
            optimizer.zero_grad()
            CEloss.backward()
            optimizer.step()
    t2 = time.time()
    times = t2-t1
    print('BS:',bs)
    print('time:',times)

"""
BS: 1
time: 8.179104089736938
BS: 2
time: 3.2590065002441406
BS: 4
time: 1.7761194705963135
BS: 8
time: 1.1629352569580078
BS: 16
time: 0.6582643985748291
"""