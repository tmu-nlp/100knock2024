import torch
import numpy as np

x_train_l = np.loadtxt("data/x_train.txt", delimiter=" ")
y_train_l = np.loadtxt("data/y_train.txt")
x_valid_l = np.loadtxt("data/x_valid.txt", delimiter=" ")
y_valid_l = np.loadtxt("data/y_valid.txt")
x_test_l = np.loadtxt("data/x_test.txt", delimiter=" ")
y_test_l = np.loadtxt("data/y_test.txt")

x_train = torch.tensor(x_train_l, dtype=torch.float32)
y_train = torch.tensor(y_train_l, dtype=torch.int64)
x_valid = torch.tensor(x_valid_l, dtype=torch.float32)
y_valid = torch.tensor(y_valid_l, dtype=torch.int64)
x_test = torch.tensor(x_test_l, dtype=torch.float32)
y_test = torch.tensor(y_test_l, dtype=torch.int64)