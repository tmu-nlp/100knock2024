'''
84. 単語ベクトルの導入Permalink
事前学習済みの単語ベクトル（例えば，Google Newsデータセット（約1,000億単語）
での学習済み単語ベクトル）で単語埋め込みemb(x)を初期化し，学習せよ．
'''

import torch
from torch import nn
from torch import cuda

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from load_and_create_dict import *
from knock81 import RNN
from knock82 import train_model, visualize_logs, calculate_loss_and_accuracy


# load model
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300_ch09.bin.gz', binary=True)

# get word2vec
VOCAB_SIZE = len(word2id) + 1
EMB_SIZE = 300
weights = np.zeros((VOCAB_SIZE, EMB_SIZE))

for i, word in enumerate(word2id.keys()):
  try:
    weights[i] = model[word] # GoogleNews-vectors-negative300
  except KeyError:
    weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE,))
weights = torch.from_numpy(weights.astype((np.float32)))


# params
VOCAB_SIZE = len(word2id) + 1
EMB_SIZE = 300
PADDING_IDX = len(word2id)
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
NUM_EPOCHS = 10

# load model
model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

# criterion
criterion = nn.CrossEntropyLoss()

# opt
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# device
device = 'cuda' if cuda.is_available() else 'cpu'

# train model
log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, NUM_EPOCHS)

# plot
visualize_logs(log)

# acc
_, acc_train = calculate_loss_and_accuracy(model, dataset_train, device)
_, acc_test = calculate_loss_and_accuracy(model, dataset_test, device)
print(f'正解率（学習データ）：{acc_train:.3f}')
print(f'正解率（評価データ）：{acc_test:.3f}')

"""
output:
epoch: 1, loss_train: 1.1116, accuracy_train: 0.5063, loss_valid: 1.1470, accuracy_valid: 0.4641, 15.9834sec
epoch: 2, loss_train: 1.0356, accuracy_train: 0.5629, loss_valid: 1.1023, accuracy_valid: 0.5127, 16.0794sec
epoch: 3, loss_train: 0.8989, accuracy_train: 0.6572, loss_valid: 0.9902, accuracy_valid: 0.6198, 16.0758sec
epoch: 4, loss_train: 0.7302, accuracy_train: 0.7382, loss_valid: 0.8449, accuracy_valid: 0.6916, 15.9716sec
epoch: 5, loss_train: 0.6390, accuracy_train: 0.7724, loss_valid: 0.7706, accuracy_valid: 0.7275, 15.8496sec
epoch: 6, loss_train: 0.5683, accuracy_train: 0.7950, loss_valid: 0.7338, accuracy_valid: 0.7395, 16.0131sec
epoch: 7, loss_train: 0.5221, accuracy_train: 0.8102, loss_valid: 0.7076, accuracy_valid: 0.7627, 16.0972sec
epoch: 8, loss_train: 0.4884, accuracy_train: 0.8204, loss_valid: 0.6928, accuracy_valid: 0.7545, 15.9836sec
epoch: 9, loss_train: 0.4701, accuracy_train: 0.8273, loss_valid: 0.6820, accuracy_valid: 0.7665, 15.8239sec
epoch: 10, loss_train: 0.4649, accuracy_train: 0.8281, loss_valid: 0.6808, accuracy_valid: 0.7650, 15.8993sec

正解率（学習データ）：0.828
正解率（評価データ）：0.779
"""