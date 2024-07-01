import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Network(nn.Module):
    def __init__(self,input_feature,output):
        super().__init__()
        self.fc1 = nn.Linear(input_feature, output)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        self.fc2 = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class NewsDataset(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.Y)

    def __getitem_(self,idx):
        return [self.X[idx],self.Y[idx]]

def calc_acc(model,loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,labels in loader:
            outputs = model(inputs)
            pred = torch.argmax(outputs,dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
        return correct/total

def calc_loss(model,criterion,loader):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss += criterion(outputs,labels).item()
            pred = torch.argmax(outputs,dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
        return loss/len(loader), correct/total 

X_train = torch.load("X_train.pt")
X_valid = torch.load("X_valid.pt")
X_test = torch.load("X_test.pt")
y_train = torch.load("y_train.pt")
y_valid = torch.load("y_valid.pt")
y_test = torch.load("y_test.pt")

dataset_train = NewsDataset(X_train, y_train)
dataset_valid = NewsDataset(X_valid, y_valid)
dataset_test = NewsDataset(X_test, y_test)

dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(
    dataset_valid, batch_size=len(dataset_valid), shuffle=False)
dataloader_test = DataLoader(
    dataset_test, batch_size=len(dataset_test), shuffle=False)

criterion = nn.CrossEntropyLoss()
l_1 = criterion(model(X_train[:1]),y_train[:1])
model.zero_grad()
l_1.backward()

l= criterion(model(X_train[:4]),y_train[:4])
model.zero_grad()
l.backward()
print(f'loss: {l:.4f}')
print(f'grad:\n{model.fc1.weight.grad}')
