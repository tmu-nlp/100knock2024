'''
73. 確率的勾配降下法による学習
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，行列W
を学習せよ．なお，学習は適当な基準で終了させればよい（例えば「100エポックで終了」など）．
'''
from tqdm import tqdm
import torch
import numpy as np
from load_vector_data import *

#network
net = torch.nn.Linear(300, 4, bias=False)
loss_fn = torch.nn.CrossEntropyLoss()

#SGD
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

#model training (epochs=100)
losses = []
for epoch in tqdm(range(1000)):
    optimizer.zero_grad()
    y_pred = torch.softmax(net.forward(x_train), dim=1)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss)

torch.save(net.state_dict(), "model.pt")