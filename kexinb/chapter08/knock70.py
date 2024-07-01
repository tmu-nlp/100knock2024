# task70. 単語ベクトルの和による特徴量

import pandas as pd
import pickle
import string
import torch


def trans_word2vec(text):
    table = str.maketrans(string.punctuation, ' ' *
                          len(string.punctuation))  # 記号を空白に
    words = text.translate(table).split()  # スペースで分割
    vec = [model[word] for word in words if word in model]  # 1語ずつベクトル化

    return torch.tensor(sum(vec) / len(vec))  # 平均ベクトルをTensor型に変換して出力


with open("output/ch7/word2vec.pkl", "rb") as f:
    model = pickle.load(f)

train = pd.read_csv("output/ch6/train.txt", sep='\t')
valid = pd.read_csv("output/ch6/valid.txt", sep='\t')
test = pd.read_csv("output/ch6/test.txt", sep='\t')

X_train = torch.stack([trans_word2vec(text) for text in train.iloc[:, 0]])
X_valid = torch.stack([trans_word2vec(text) for text in valid.iloc[:, 0]])
X_test = torch.stack([trans_word2vec(text) for text in test.iloc[:, 0]])

category = {'b': 0, 't': 1, 'e': 2, 'm': 3}
y_train = torch.tensor(train.iloc[:, 1].map(lambda x: category[x]).values)
y_valid = torch.tensor(valid.iloc[:, 1].map(lambda x: category[x]).values)
y_test = torch.tensor(test.iloc[:, 1].map(lambda x: category[x]).values)

torch.save(X_train, 'output/ch8/X_train.pt')
torch.save(X_valid, 'output/ch8/X_valid.pt')
torch.save(X_test, 'output/ch8/X_test.pt')
torch.save(y_train, 'output/ch8/y_train.pt')
torch.save(y_valid, 'output/ch8/y_valid.pt')
torch.save(y_test, 'output/ch8/y_test.pt')

# print(X_train.size())
# print(X_train)
# print(y_train.size())
# print(y_train)

'''
torch.Size([10671, 300])
tensor([[ 0.0538,  0.0353,  0.0121,  ...,  0.1317,  0.2012,  0.0249],
        [ 0.1075,  0.0263, -0.0342,  ...,  0.0381,  0.0845,  0.0992],
        [-0.0859,  0.0667, -0.0921,  ..., -0.0673,  0.1223, -0.0711],
        ...,
        [ 0.0018,  0.0506,  0.0403,  ..., -0.0771,  0.0192, -0.0621],
        [-0.0252,  0.0566, -0.0808,  ...,  0.0292,  0.0887,  0.0360],
        [-0.0016,  0.1155, -0.0469,  ...,  0.0594, -0.0317, -0.1071]])
torch.Size([10671])
tensor([3, 2, 0,  ..., 2, 0, 0])
'''