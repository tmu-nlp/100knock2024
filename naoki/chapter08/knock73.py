import pandas as pd
import numpy as np
import torch
import torch.nn as nn

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

loss = nn.CrossEntropyLoss(reduction='mean')
X_train = torch.load('X_train.pt')
Y_train = torch.load('y_train.pt')
model = Network(300,4)
Y_pred = model(X_train)
CEloss = loss(Y_pred,Y_train)

model.zero_grad()
CEloss.backward()
print(model.fc1.weight.grad)