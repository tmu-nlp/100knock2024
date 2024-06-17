'''
72. 損失と勾配の計算
学習データの事例x1と事例集合x1,x2,x3,x4に対して，
クロスエントロピー損失と，行列Wに対する勾配を計算せよ
'''
import torch
import numpy as np
from load_vector_data import *

#network
net = torch.nn.Linear(300, 4, bias=False)

#forward
y_pred1 = torch.softmax(net.forward(x_train[:1]), dim=1)
y_pred4 = torch.softmax(net.forward(x_train[:4]), dim=1)

#cross entropy loss
loss = torch.nn.CrossEntropyLoss()
loss1 = loss(y_pred1, y_train[:1])
loss4 = loss(y_pred4, y_train[:4])

#backward propagation
#id 1
net.zero_grad()#gradient reset 
loss1.backward()
print(f"id 1 CE loss: {loss1}")
print(f"id 1 gradient: {net.weight.grad}")

#id 1 to 4
net.zero_grad()#gradient reset
loss4.backward()
print(f"id 1~4 CE loss: {loss4}")
print(f"id 1~4 gradient: {net.weight.grad}")