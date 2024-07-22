#81
import re
import pandas as pd
import torch
from collections import defaultdict
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
    # !?以外の記号の削除
    text = remove_marks_regex.sub("", text)
    # !?と単語の間にスペースを挿入
    text = shift_marks_regex.sub(r" \1 ", text)
    return text

class RNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, emb_size, pad_idx, output_size):
        super().__init__()
        self.hid_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity="tanh", batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = torch.zeros(1, self.batch_size, self.hid_size)
        emb = self.emb(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.fc(out[:, -1, :])
        return out

class TITLEDataset(Dataset):
    def __init__(self):
        # ファイルの読み込み
        header_name = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
        X_train = pd.read_table('drive/MyDrive/df_train.txt', header=None, names=header_name)

        # 辞書の作成
        d = defaultdict(int)
        for sentence in X_train['TITLE']:
            for word in cleanText(sentence).split():
                d[word] += 1 # それぞれの単語が出現するごとにカウントを増やす
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
        category = {'b': 0, 't': 1, 'e': 2, 'm': 3}
        self.labels = torch.tensor(X_train['CATEGORY'].map(lambda x: category[x]).values)

    @property
    def vocab_size(self):
        return len(self.word2token)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data, n_tokens = self.data[idx]
        label = self.labels[idx]
        return data, label, n_tokens

if __name__ == "__main__":
    train_data = TITLEDataset()
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, num_workers=4)

    VOCAB_SIZE = train_data.vocab_size + 1
    EMB_SIZE = 300
    PADDING_IDX = train_data.vocab_size
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50

    model = RNN(HIDDEN_SIZE, VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE)

    for epoch in range(1):
        model.train()
        for x, y, nt in train_loader:
            y_pred = model(x)
            y_pred = torch.softmax(y_pred, dim=-1)
            print(y_pred)