# task72. 損失と勾配の計算

from torch import nn
import torch
from knock71 import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class NewsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]


def calc_acc(model, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

        return correct / total


def calc_loss_acc(model, criterion, loader):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

        return loss / len(loader), correct / total


X_train = torch.load("output/ch8/X_train.pt")
X_valid = torch.load("output/ch8/X_valid.pt")
X_test = torch.load("output/ch8/X_test.pt")
y_train = torch.load("output/ch8/y_train.pt")
y_valid = torch.load("output/ch8/y_valid.pt")
y_test = torch.load("output/ch8/y_test.pt")

dataset_train = NewsDataset(X_train, y_train)
dataset_valid = NewsDataset(X_valid, y_valid)
dataset_test = NewsDataset(X_test, y_test)

dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(
    dataset_valid, batch_size=len(dataset_valid), shuffle=False)
dataloader_test = DataLoader(
    dataset_test, batch_size=len(dataset_test), shuffle=False)

criterion = nn.CrossEntropyLoss()
l_1 = criterion(model(X_train[:1]), y_train[:1])
model.zero_grad()
l_1.backward()
# print(f'loss: {l_1:.4f}')
# print(f'grad:\n{model.fc.weight.grad}')

l = criterion(model(X_train[:4]), y_train[:4])
model.zero_grad()
l.backward()
# print(f'loss: {l:.4f}')
# print(f'grad:\n{model.fc.weight.grad}')

'''
loss: 0.5878
grad:
tensor([[ 0.0024,  0.0016,  0.0005,  ...,  0.0060,  0.0091,  0.0011],
        [ 0.0207,  0.0136,  0.0047,  ...,  0.0507,  0.0774,  0.0096],
        [ 0.0008,  0.0005,  0.0002,  ...,  0.0019,  0.0029,  0.0004],
        [-0.0239, -0.0157, -0.0054,  ..., -0.0585, -0.0894, -0.0111]])
loss: 1.8198
grad:
tensor([[ 0.0464, -0.0075,  0.0181,  ...,  0.0206, -0.0115,  0.0300],
        [ 0.0050,  0.0069, -0.0031,  ...,  0.0107,  0.0263,  0.0017],
        [-0.0493, -0.0126,  0.0025,  ..., -0.0054, -0.0211, -0.0221],
        [-0.0021,  0.0132, -0.0174,  ..., -0.0258,  0.0064, -0.0096]])
'''