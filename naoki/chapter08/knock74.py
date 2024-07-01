import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

class Network(nn.Module):
    def __init__(self,input_feature,output):
        super().__init__()
        self.fc1 = nn.Linear(input_feature, output)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        self.fc2 = nn.Softmax(dim=1)
    def forward(self,x):
        y = self.fc1(x)
        z = self.fc2(y)
        return z

loss = nn.CrossEntropyLoss(reduction='mean')
X_train = torch.load('X_train.pt')
y_train = torch.load('y_train.pt')

#model
from sklearn.metrics import accuracy_score
model = Network(300, 4)
model.load_state_dict(torch.load("model.pt"))

_, y_pred_train = torch.max(model(X_train),dim=1)
print(f"train data acc:{accuracy_score(y_pred_train, y_train)}")
_, y_pred_test = torch.max(model(X_test),dim=1)
print(f"train data acc:{accuracy_score(y_pred_test, y_test)}")