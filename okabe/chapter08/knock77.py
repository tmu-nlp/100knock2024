'''
77. ミニバッチ化Permalink
問題76のコードを改変し，B事例ごとに損失・勾配を計算し，行列Wの値を更新せよ
（ミニバッチ化）．Bの値を1,2,4,8,…と変化させながら，1エポックの学習に要する時間を比較せよ．
'''
from tqdm import tqdm
import torch, time
import numpy as np
from load_vector_data import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(seed=0)

#network
net = torch.nn.Linear(300, 4, bias=False)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

batchsize = [2**i for i in range(5)]

#dataset
dataset = TensorDataset(x_train, y_train)

times = []

for bs in batchsize:
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoch in tqdm(range(10)):
        start = time.time() #timer start
        for xx, yy in loader:
            optimizer.zero_grad()
            y_pred = net(x_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

        '''
        net.eval()
        #loss
        train_losses.append(loss.detach().numpy())
        valid_losses.append(loss_fn(net(x_valid), y_valid).detach().numpy())
        #pred
        y_max_train, y_pred_train = torch.max(net(x_train),dim=1)
        y_max_valid, y_pred_valid = torch.max(net(x_valid),dim=1)
        #acc
        train_acc = accuracy_score(y_pred_train, y_train)
        valid_acc = accuracy_score(y_pred_valid, y_valid)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        '''

        #timer end
        times.append(time.time() - start)

#time record
with open ("output/time.txt", "w") as f:
    for i in range(len(times)):
        if i%10 == 0:
            print(f"---------bachsize={2**(i/10)}----------", file=f)
        print(f"epoch{i%10+1}:{times[i]}", file=f)

'''
output:
100%|██████████| 10/10 [05:08<00:00, 30.80s/it]
100%|██████████| 10/10 [02:52<00:00, 17.21s/it]
100%|██████████| 10/10 [01:25<00:00,  8.58s/it]
100%|██████████| 10/10 [00:43<00:00,  4.33s/it]
100%|██████████| 10/10 [00:22<00:00,  2.25s/it]
'''