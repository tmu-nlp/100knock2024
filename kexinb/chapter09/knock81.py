# task81. RNNによる予測

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from knock80 import *

# Define RNN model
class RNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, emb_size, pad_idx, output_size):
        super().__init__()
        self.hid_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity="tanh", batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = torch.zeros(1, batch_size, self.hid_size, device=x.device)
        emb = self.emb(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.fc(out[:, -1, :])
        return out

# Define custom dataset
class NewsDataset(Dataset):
    def __init__(self, x, y, tokenizer):
        self.x = x
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        text = self.x[idx]
        inputs = self.tokenizer(text)
        return {
            'inputs': torch.tensor(inputs, dtype=torch.int64),
            'labels': self.y[idx]
        }

# Load and preprocess data
def load_data(file_path):
    header_name = ['TITLE', 'CATEGORY']
    data = pd.read_csv(file_path, header=None, sep='\t', names=header_name)
    return data

def preprocess_labels(data, category_dict):
    return torch.tensor(data['CATEGORY'].map(lambda x: category_dict[x]).values, dtype=torch.int64)

# Load data
train_file = "output/ch6/train.txt"
valid_file = "output/ch6/valid.txt"
test_file = "output/ch6/test.txt"
category = {'b': 0, 't': 1, 'e': 2, 'm': 3}

train_data = load_data(train_file)
valid_data = load_data(valid_file)
test_data = load_data(test_file)

y_train = preprocess_labels(train_data, category)
y_valid = preprocess_labels(valid_data, category)
y_test = preprocess_labels(test_data, category)

dataset_train = NewsDataset(train_data["TITLE"], y_train, word_ids.return_id)
dataset_valid = NewsDataset(valid_data["TITLE"], y_valid, word_ids.return_id)
dataset_test = NewsDataset(test_data["TITLE"], y_test, word_ids.return_id)

train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset_valid, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

# Main processing
if __name__ == "__main__":
    VOCAB_SIZE = len(set(word_ids.id_dict.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word_ids.id_dict.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50

    model = RNN(HIDDEN_SIZE, VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE)
    print(f'len(Dataset): {len(dataset_train)}')
    # tokenized input tensor & corresponding label
    print('Dataset[index]:')
    for var in dataset_train[0]:
        print(f'  {var}: {dataset_train[0][var]}')
    # predicted probabilities for first 10 instances
    for i in range(10):
        X = dataset_train[i]['inputs']
        print(torch.softmax(model(X.unsqueeze(0)), dim=-1))

'''
len(Dataset): 10672
Dataset[0]:
  inputs: tensor([ 171, 6636,   62, 6637,   23, 3531, 3532,   11, 1067, 2328])
  labels: 0
tensor([[0.4963, 0.1731, 0.2046, 0.1260]], grad_fn=<SoftmaxBackward0>)
tensor([[0.3680, 0.3086, 0.2416, 0.0818]], grad_fn=<SoftmaxBackward0>)
tensor([[0.2895, 0.1695, 0.2958, 0.2452]], grad_fn=<SoftmaxBackward0>)
tensor([[0.3525, 0.1549, 0.1428, 0.3498]], grad_fn=<SoftmaxBackward0>)
tensor([[0.4358, 0.2188, 0.1659, 0.1795]], grad_fn=<SoftmaxBackward0>)
tensor([[0.1825, 0.2260, 0.2954, 0.2961]], grad_fn=<SoftmaxBackward0>)
tensor([[0.3637, 0.1999, 0.2539, 0.1825]], grad_fn=<SoftmaxBackward0>)
tensor([[0.2179, 0.2251, 0.1294, 0.4276]], grad_fn=<SoftmaxBackward0>)
tensor([[0.4316, 0.1664, 0.2620, 0.1399]], grad_fn=<SoftmaxBackward0>)
tensor([[0.0838, 0.2111, 0.2920, 0.4131]], grad_fn=<SoftmaxBackward0>)
'''