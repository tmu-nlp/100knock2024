'''
82. 確率的勾配降下法による学習Permalink
確率的勾配降下法（SGD: Stochastic Gradient Descent）を用いて，
問題81で構築したモデルを学習せよ．
訓練データ上の損失と正解率，評価データ上の損失と正解率を表示しながらモデルを学習し，
適当な基準（例えば10エポックなど）で終了させよ．
'''
import time

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_and_create_dict import *
from knock81 import RNN

def calc_loss_acc(model, dataset, device="cuda", criterion=None):
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
  loss = 0.0
  total = 0
  correct = 0
  with torch.no_grad():
    for data in dataloader:
      inputs = data['inputs'].to(device)
      labels = data['labels'].to(device)

      #forward
      outputs = model(inputs)

      #loss
      if criterion != None:
        loss += criterion(outputs, labels).item()

      #acc
      pred = torch.argmax(outputs, dim=-1)
      total += len(inputs)
      correct += (pred == labels).sum().item()

  return loss / len(dataset), correct / total


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, collate_fn=None, device="cuda"):
  model.to(device)

  # dataloader
  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
  dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)

  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)

  #train
  log_train = []
  log_valid = []
  for epoch in range(num_epochs):
    s_time = time.time()
    model.train()
    for data in dataloader_train:
      optimizer.zero_grad()

      # forward + back prop + optim
      inputs = data['inputs'].to(device)
      labels = data['labels'].to(device)
      outputs = model.forward(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    
    # eval
    model.eval()
    loss_train, acc_train = calc_loss_acc(model, dataset_train, device, criterion=criterion)
    loss_valid, acc_valid = calc_loss_acc(model, dataset_valid, device, criterion=criterion)
    log_train.append([loss_train, acc_train])
    log_valid.append([loss_valid, acc_valid])

    #checkpoint
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')

    e_time = time.time()
    print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec')

    # end if val_loss doesn't decrease for 3 epochs
    if epoch > 2 and log_valid[epoch - 3][0] <= log_valid[epoch - 2][0] <= log_valid[epoch - 1][0] <= log_valid[epoch][0]:
      break

    scheduler.step()

  return {'train': log_train, 'valid': log_valid}

def visualize_logs(log):
  _, ax = plt.subplots(1, 2, figsize=(15, 5))
  ax[0].plot(np.array(log['train']).T[0], label='train')
  ax[0].plot(np.array(log['valid']).T[0], label='valid')
  ax[0].set_xlabel('epoch')
  ax[0].set_ylabel('loss')
  ax[0].legend()
  ax[1].plot(np.array(log['train']).T[1], label='train')
  ax[1].plot(np.array(log['valid']).T[1], label='valid')
  ax[1].set_xlabel('epoch')
  ax[1].set_ylabel('accuracy')
  ax[1].legend()
  plt.show()

# param
VOCAB_SIZE = len(set(word2id.values())) + 1  
EMB_SIZE = 300
PADDING_IDX = len(set(word2id.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
NUM_EPOCHS = 10

# load model
model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

# criterion
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# train model
log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS)

# plot
visualize_logs(log)

# acc
_, acc_train = calc_loss_acc(model, dataset_train)
_, acc_test = calc_loss_acc(model, dataset_test)
print(f'正解率（学習データ）：{acc_train:.3f}')
print(f'正解率（評価データ）：{acc_test:.3f}')

"""
output:
epoch: 1, loss_train: 1.1092, accuracy_train: 0.5098, loss_valid: 1.1275, accuracy_valid: 0.5120, 16.1852sec
epoch: 2, loss_train: 1.0325, accuracy_train: 0.5714, loss_valid: 1.0758, accuracy_valid: 0.5494, 17.1496sec
epoch: 3, loss_train: 0.9091, accuracy_train: 0.6536, loss_valid: 0.9818, accuracy_valid: 0.6287, 15.8530sec
epoch: 4, loss_train: 0.7495, accuracy_train: 0.7322, loss_valid: 0.8575, accuracy_valid: 0.6984, 16.0288sec
epoch: 5, loss_train: 0.6696, accuracy_train: 0.7598, loss_valid: 0.8090, accuracy_valid: 0.7208, 15.9361sec
epoch: 6, loss_train: 0.5970, accuracy_train: 0.7858, loss_valid: 0.7530, accuracy_valid: 0.7410, 16.4367sec
epoch: 7, loss_train: 0.5463, accuracy_train: 0.8020, loss_valid: 0.7290, accuracy_valid: 0.7410, 15.7477sec
epoch: 8, loss_train: 0.5091, accuracy_train: 0.8144, loss_valid: 0.6953, accuracy_valid: 0.7515, 15.8105sec
epoch: 9, loss_train: 0.4909, accuracy_train: 0.8171, loss_valid: 0.6823, accuracy_valid: 0.7582, 15.9419sec
epoch: 10, loss_train: 0.4867, accuracy_train: 0.8189, loss_valid: 0.6812, accuracy_valid: 0.7590, 15.7161sec

正解率（学習データ）：0.819
正解率（評価データ）：0.764
"""