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

X_train = torch.load('X_train.pt')
model = Network(300,4)
y_hat_1 = model(X_train[:1])
Y_hat = model(X_train[:4])
y_hat_1, Y_hat