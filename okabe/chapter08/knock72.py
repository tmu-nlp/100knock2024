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

"""
output:
id 1 CE loss: 1.383530616760254
id 1 gradient: tensor([[-0.0173,  0.0280,  0.0020,  ...,  0.0441, -0.0262, -0.0411],
        [ 0.0057, -0.0093, -0.0007,  ..., -0.0146,  0.0087,  0.0136],
        [ 0.0053, -0.0085, -0.0006,  ..., -0.0134,  0.0080,  0.0125],
        [ 0.0063, -0.0102, -0.0007,  ..., -0.0161,  0.0095,  0.0150]])
id 1~4 CE loss: 1.382717251777649
id 1~4 gradient: tensor([[-6.1116e-03, -3.2874e-03,  7.6965e-03,  ...,  1.1390e-02,
         -1.8714e-02, -1.0351e-02],
        [ 1.3105e-03,  1.7909e-03, -3.1221e-03,  ..., -6.0536e-03,
          6.8856e-03,  3.1020e-03],
        [ 3.3788e-03, -9.1946e-05, -1.2998e-03,  ...,  1.3072e-03,
          4.7228e-03,  3.7497e-03],
        [ 1.4223e-03,  1.5884e-03, -3.2746e-03,  ..., -6.6438e-03,
          7.1055e-03,  3.4992e-03]])
"""