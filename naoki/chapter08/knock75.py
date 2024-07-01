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
loss = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.05)

loss_train = []
acc_train = []
loss_valid = []
acc_valid = []

for epoc in range(100000):
    optimizer.zero_grad()
    y_pred_train = model(X_train)
    CEloss_train = loss(y_pred_train, y_train)
    CEloss_train.backward()
    optimizer.step()
    with torch.no_grad():
        y_pred_valid = model(X_valid)
        CEloss_valid = loss(y_pred_valid, y_valid)
    loss_train.append(CEloss_train.item())
    loss_valid.append(CEloss_valid.item())

    _, y_pred_train = torch.max(model(X_train),dim=1)
    acc_train.append(accuracy_score(y_pred_train.detach().numpy(), y_train.detach().numpy()))
    _, y_pred_valid = torch.max(model(X_valid),dim=1)
    acc_valid.append(accuracy_score(y_pred_valid.detach().numpy(), y_valid.detach().numpy()))

#loss plot
fig = plt.figure()
plt.plot(loss_train, label="train loss")
plt.plot(loss_valid, label="valid loss")
plt.legend()

#accuracy plot
fig = plt.figure()
plt.plot(acc_train, label="train acc")
plt.plot(acc_valid, label="valid acc")
plt.legend()

