import torch
import numpy as np
from load_vector_data import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0)

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#network
net = torch.nn.Linear(300, 4, bias=False)
net = net.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

batchsize = [2**i for i in range(5)]

#dataset
dataset = TensorDataset(x_train.to(device), y_train.to(device))

times = []

for bs in batchsize:
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoch in tqdm(range(10)):
        start = time.time()#timer start
        for xx, yy in loader:
            xx, yy = xx.to(device), yy.to(device)
            optimizer.zero_grad()
            y_pred = net(xx)
            loss = loss_fn(y_pred, yy)
            loss.backward()
            optimizer.step()
        
        '''
        model.eval()
        #損失の記録
        train_losses.append(loss.detach().numpy())
        valid_losses.append(loss_fn(net(x_valid), y_valid).detach().numpy())
        #カテゴリの予測
        y_max_train, y_pred_train = torch.max(net(x_train),dim=1)
        y_max_valid, y_pred_valid = torch.max(net(x_valid),dim=1)
        #正解率の記録
        train_acc = accuracy_score(y_pred_train, y_train)
        valid_acc = accuracy_score(y_pred_valid, y_valid)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        '''

        #timer end
        times.append(time.time() - start)

        
#timer record
with open ("output/time_gpu.txt", "w") as f:
    for i in range(len(times)):
        if i%10 == 0:
            print(f"---------bachsize={2**(i/10)}----------", file=f)
        print(f"epoch{i%10+1}:{times[i]}", file=f)

'''
output:
cuda
100%|██████████| 10/10 [00:31<00:00,  3.15s/it]
100%|██████████| 10/10 [00:14<00:00,  1.44s/it]
100%|██████████| 10/10 [00:07<00:00,  1.30it/s]
100%|██████████| 10/10 [00:04<00:00,  2.42it/s]
100%|██████████| 10/10 [00:02<00:00,  4.42it/s]
'''