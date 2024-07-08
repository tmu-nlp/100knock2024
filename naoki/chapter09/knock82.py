#82
import re
from collections import defaultdict
import joblib
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def list2tensor(token_idxes, max_len=100, padding=True):
    if len(token_idxes) > max_len:
        token_idxes = token_idxes[:max_len]
    n_tokens = len(token_idxes)
    if padding:
        token_idxes = token_idxes + [0] * (max_len - len(token_idxes))
    return torch.tensor(token_idxes, dtype=torch.int64), n_tokens

def cleanText(text):
    remove_marks_regex = re.compile("[,\.\(\)\[\]\*:;]|<.*?>")
    shift_marks_regex = re.compile("([?!])")
    # !?以外の記号の削除
    text = remove_marks_regex.sub("", text)
    # !?と単語の間にスペースを挿入
    text = shift_marks_regex.sub(r" \1 ", text)
    return text

class RNN(nn.Module):
    def __init__(self, num_embeddings,
                 embedding_dim,
                 hidden_size,
                 output_size,
                 num_layers, #RNNは重ねることが可能
                 dropout=0.2): #過学習の防止
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim,
                                padding_idx=0)
        self.rnn = nn.RNN(embedding_dim,
                          hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = torch.zeros(1, self.batch_size, self.hidden_size)   #初期状態
        # IDをEmbeddingで多次元のベクトルに変換する
        # (batch_size, step_size) -> (batch_size, step_size, embedding_dim)
        emb = self.emb(x)
        # (batch_size, step_size, embedding_dim) -> (batch_size, step_size, hidden_dim)
        out, hidden = self.rnn(emb, hidden)
        out = self.linear(out[:, -1, :])
        return out

class TITLEDataset(Dataset):
    def __init__(self,section):
        #ファイルの読み込み
        header_name = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
        X_train = pd.read_table(f'drive/MyDrive/{section}.txt', header=None, names=header_name)

        #辞書の作成
        d = defaultdict(int)
        for sentence in X_train['TITLE']:
            for word in sentence.split():
                d[word] += 1 #それぞれの単語が出現するごとにカウントを増やす
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

        self.data = (X_train['TITLE'].apply(lambda x: list2tensor([self.word2token.get(word, 0) for word in cleanText(x).split()])))

        y = pd.read_table(f'drive/MyDrive/{section}.txt', header=None)[4].values
        # Convert string labels to numerical labels
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

def eval_net(net, data_loader):
    net.eval()
    ys = []
    ypreds = []
    for x, y, nt in data_loader:
        with torch.no_grad():
            y_pred = net(x)
            y_pred = torch.softmax(y_pred,dim=-1)
            print(f'test loss: {loss_fn(y_pred, y.long()).item()}')
            _, y_pred = torch.max(y_pred, 1)
            ys.append(y)
            ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    print(f'test acc: {(ys == ypreds).sum().item() / len(ys)}')
    return

if __name__ == "__main__":
    train_data = TITLEDataset(section='df_train')
    train_loader = DataLoader(train_data, batch_size=len(train_data),
                            shuffle=True, num_workers=4)
    valid_data = TITLEDataset(section='df_valid')
    valid_loader = DataLoader(valid_data, batch_size=len(valid_data),
                            shuffle=False, num_workers=4)

    net = RNN(train_data.vocab_size + 1,embedding_dim=50,hidden_size=100, num_layers=1, output_size=4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in tqdm(range(10)):
        losses = []
        net.train()
        for x, y, nt in train_loader:
            y_pred = net(x)
            y_pred = torch.softmax(y_pred,dim=-1)
            loss = loss_fn(y_pred, y.long())
            net.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            _, y_pred_train = torch.max(y_pred, 1)
            print(f'train loss: {loss.item()}')
            print(f'train acc: {(y_pred_train == y).sum().item() / len(y)}')
        eval_net(net, valid_loader)