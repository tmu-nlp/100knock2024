import torch
from torch import nn

class MyRNN(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
    super(MyRNN, self).__init__()

    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.rnn = nn.RNN(embedding_dim, hidden_dim)
    self.fc = nn.Linear(hidden_dim, output_size)

  def forward(self, x):
    x = self.embeddings(x)
    x, h_T = self.rnn(x)
    x = x[:, -1, :]
    x = self.fc(x)
    pred = torch.softmax(x, dim=-1)
    return pred