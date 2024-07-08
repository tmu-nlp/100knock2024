import re
from collections import defaultdict
import joblib
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

def list2tensor(token_idxes, max_len=20, padding=True):
    if len(token_idxes) > max_len:
        token_idxes = token_idxes[:max_len]
    n_tokens = len(token_idxes)
    if padding:
        token_idxes = token_idxes + [0] * (max_len - len(token_idxes))
    return torch.tensor(token_idxes, dtype=torch.int64), n_tokens

def cleanText(text):
    remove_marks_regex = re.compile("[,\.\(\)\[\]\*:;]|<.*?>")
    shift_marks_regex = re.compile("([?!])")
    text = remove_marks_regex.sub("", text)
    text = shift_marks_regex.sub(r" \1 ", text)
    return text

class RNN(nn.Module):
    def __init__(self, num_embeddings,
                 embedding_dim,
                 hidden_size,
                 output_size,
                 num_layers, 
                 dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.batch_size = x.size(0)
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(x.device)
        emb = self.emb(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.linear(out[:, -1, :])
        return out

class TITLEDataset(Dataset):
    def __init__(self, section):
        header_name = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
        X_train = pd.read_table(f'drive/MyDrive/{section}.txt', header=None, names=header_name)
        
        d = defaultdict(int)
        for sentence in X_train['TITLE']:
            for word in sentence.split():
                d[word] += 1
        dc = sorted(d.items(), key=lambda x: x[1], reverse=True)

        words = []
        idx = []
        for i, word_dic in enumerate(dc, 1):
            words.append(word_dic[0])
            if word_dic[1] < 2:
                idx.append(0)
            else:
                idx.append(i)
        self.word2token = dict(zip(words, idx))

        self.data = X_train['TITLE'].apply(lambda x: list2tensor([self.word2token.get(word, 0) for word in cleanText(x).split()]))
        y = X_train['CATEGORY'].values
        unique_labels = np.unique(y)
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = np.array([self.label2idx[label] for label in y])

    @property
    def vocab_size(self):
        return len(self.word2token)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data, n_tokens = self.data[idx]
        label = self.labels[idx]
        return data, label, n_tokens

def eval_net(net, data_loader, loss_fn, device='cpu'):
    net.eval()
    ys = []
    ypreds = []
    losses = []
    with torch.no_grad():
        for x, y, nt in data_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = net(x)
            loss = loss_fn(y_pred, y)
            losses.append(loss.item())
            _, y_pred = torch.max(y_pred, 1)
            ys.append(y)
            ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    acc = (ys == ypreds).sum().item() / len(ys)
    return np.mean(losses), acc

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data = TITLEDataset(section='df_train')
    batch_size = 640
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_data = TITLEDataset(section='df_valid')
    valid_loader = DataLoader(valid_data, batch_size=len(valid_data), shuffle=False, num_workers=4)

    net = RNN(train_data.vocab_size + 1, embedding_dim=300, hidden_size=300, num_layers=1, output_size=4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in tqdm(range(10)):
        net.train()  # モデルをトレーニングモードに設定
        total_loss = 0
        for x, y, nt in train_loader:
            x = x.to(device)  # 入力データをデバイスに転送
            y = y.to(device)  # ラベルデータをデバイスに転送
            y_pred = net(x)  # モデルで予測を計算
            loss = loss_fn(y_pred, y)  # 損失を計算
            optimizer.zero_grad()  # 勾配をリセット
            loss.backward()  # 勾配を計算
            optimizer.step()  # パラメータを更新
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        val_loss, val_acc = eval_net(net, valid_loader, loss_fn, device)
        print(f'Epoch {epoch+1}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')