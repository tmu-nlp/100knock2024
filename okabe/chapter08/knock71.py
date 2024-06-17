'''
71. 単層ニューラルネットワークによる予測
'''
import torch
import numpy as np
from load_vector_data import *

W = torch.rand(300, 4)

softmax = torch.nn.Softmax(dim=1)
print(softmax(torch.matmul(x_train[:1], W)))
print(softmax(torch.matmul(x_train[:4], W)))