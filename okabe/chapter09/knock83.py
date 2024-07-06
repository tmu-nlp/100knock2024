'''
83. ミニバッチ化・GPU上での学習Permalink
問題82のコードを改変し，B事例ごとに損失・勾配を計算して学習を行えるようにせよ
（Bの値は適当に選べ）．また，GPU上で学習を実行せよ．
'''

import torch
from torch import nn
from torch import cuda

import pandas as pd

from load_and_create_dict import *
from knock81 import RNN, CreateDataset
from knock82 import train_model, visualize_logs, calc_loss_acc



class Padsequence():
  def __init__(self, padding_idx):
    self.padding_idx = padding_idx

  def __call__(self, batch):
    sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
    sequences = [x['inputs'] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
    labels = torch.LongTensor([x['labels'] for x in sorted_batch])

    return {'inputs': sequences_padded, 'labels': labels}


# params
VOCAB_SIZE = len(set(word2id.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(word2id.values()))
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50
LEARNING_RATE = 5e-2
BATCH_SIZE = 32
NUM_EPOCHS = 10

# モデルの定義
model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

# criterion
criterion = nn.CrossEntropyLoss()

# opt
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# device
device = 'cuda' if cuda.is_available() else 'cpu'

# train model
log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS, collate_fn=Padsequence(PADDING_IDX), device=device)

# plot
visualize_logs(log)

# acc
_, acc_train = calc_loss_acc(model, dataset_train, device)
_, acc_test = calc_loss_acc(model, dataset_test, device)
print(f'正解率（学習データ）：{acc_train:.3f}')
print(f'正解率（評価データ）：{acc_test:.3f}')

"""
output:
epoch: 1, loss_train: 1.2511, accuracy_train: 0.3928, loss_valid: 1.2417, accuracy_valid: 0.4042, 6.4499sec
epoch: 2, loss_train: 1.2378, accuracy_train: 0.4235, loss_valid: 1.2478, accuracy_valid: 0.4199, 6.0976sec
epoch: 3, loss_train: 1.1606, accuracy_train: 0.5039, loss_valid: 1.1669, accuracy_valid: 0.4850, 6.0038sec
epoch: 4, loss_train: 1.0609, accuracy_train: 0.5963, loss_valid: 1.0795, accuracy_valid: 0.5801, 5.9414sec
epoch: 5, loss_train: 0.9686, accuracy_train: 0.6673, loss_valid: 1.0176, accuracy_valid: 0.6265, 5.9135sec
epoch: 6, loss_train: 0.8740, accuracy_train: 0.7190, loss_valid: 0.9223, accuracy_valid: 0.6886, 5.9813sec
epoch: 7, loss_train: 0.8320, accuracy_train: 0.7306, loss_valid: 0.9089, accuracy_valid: 0.6826, 5.8938sec
epoch: 8, loss_train: 0.7647, accuracy_train: 0.7573, loss_valid: 0.8409, accuracy_valid: 0.7193, 5.9531sec
epoch: 9, loss_train: 0.7355, accuracy_train: 0.7656, loss_valid: 0.8173, accuracy_valid: 0.7275, 5.9013sec
epoch: 10, loss_train: 0.7247, accuracy_train: 0.7680, loss_valid: 0.8096, accuracy_valid: 0.7290, 5.8728sec

正解率（学習データ）：0.768
正解率（評価データ）：0.738
"""