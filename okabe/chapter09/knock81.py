'''
81. RNNによる予測Permalink
再帰型ニューラルネットワーク（RNN: Recurrent Neural Network）を用い，
単語列xからカテゴリyを予測するモデルとして，次式を実装せよ．
'''
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from knock80 import *

class RNN(nn.Module): 
  def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
    self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity='tanh', batch_first=True) # batch_firstをTrueにすると，(seq_len, batch, input_size)と指定されている入力テンソルの型を(batch, seq_len, input_size)にできる
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    self.batch_size = x.size()[0]
    hidden = self.init_hidden(x.device)  
    emb = self.emb(x)
    # emb.size() : (batch_size, seq_len, emb_size)
    out, hidden = self.rnn(emb, hidden)
    # out.size() : (batch_size, seq_len, hidden_size)
    out = self.fc(out[:, -1, :])
    # out.size() : (batch_size, output_size)
    return out

  def init_hidden(self, device):
    hidden = torch.zeros(1, self.batch_size, self.hidden_size, device=device)
    return hidden


class CreateDataset(Dataset):
  def __init__(self, X, y, tokenizer):
    self.X = X
    self.y = y
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.y)

  def __getitem__(self, index):
    text = self.X[index]
    inputs = self.tokenizer(text, word2id)

    return {
      'inputs': torch.tensor(inputs, dtype=torch.int64),
      'labels': torch.tensor(self.y[index], dtype=torch.int64)
    }

#label
category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = train['CATEGORY'].map(lambda x: category_dict[x]).values
y_valid = valid['CATEGORY'].map(lambda x: category_dict[x]).values
y_test = test['CATEGORY'].map(lambda x: category_dict[x]).values


#parameters
VOCAB_SIZE = len(word2id) + 1
EMB_SIZE = 300
PADDING_IDX = len(word2id)
OUTPUT_SIZE = 4
HIDDEN_SIZE = 50

# Dataset
dataset_train = CreateDataset(train['TITLE'], y_train, tokenizer)
dataset_valid = CreateDataset(valid['TITLE'], y_valid, tokenizer)
dataset_test = CreateDataset(test['TITLE'], y_test, tokenizer)

#load model
RNN_model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

#test for the first 3 elms
for i in range(3):
    X = dataset_train[i]['inputs']
    print(torch.softmax(RNN_model(X.unsqueeze(0)), dim=-1))

"""
output:
tensor([[0.1775, 0.2947, 0.2769, 0.2509]], grad_fn=<SoftmaxBackward0>)
tensor([[0.2923, 0.2284, 0.3094, 0.1699]], grad_fn=<SoftmaxBackward0>)
tensor([[0.3504, 0.1651, 0.3536, 0.1309]], grad_fn=<SoftmaxBackward0>)
"""