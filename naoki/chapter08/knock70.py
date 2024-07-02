import pandas as pd
import numpy as np
import sklearn
import pickle
import gensim
import string
import torch
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

#データに名前づけ
header_name = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
df = pd.read_csv('drive/MyDrive/newsCorpora.csv', header=None, sep='\t', names=header_name)
"""
情報源（publisher）が
”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”
の事例（記事）のみを抽出
"""
#抽出
df_new = df[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]

#データの分割
df_train, df_other = train_test_split(
    df_new, test_size=0.2, random_state=7)
df_valid, df_test = train_test_split(
    df_other, test_size=0.5, random_state=7)

#記号を空白に置き換えた後、単語をベクトル化し、テンソル変換
def trans(text):
  for char in string.punctuation:
    text = text.replace(char, ' ')
    words = text.split()
    vec = []
    for word in words:
      if word in model:
        vec.append(model[word])

    if len(vec)>0:
      ave = np.mean(vec, axis=0)
      return torch.tensor(ave)
    else:
      return torch.zeros(model.vector_size)

model = KeyedVectors.load_word2vec_format("drive/MyDrive/GoogleNews-vectors-negative300.bin.gz", binary=True)

with open("drive/MyDrive/word2vec.pkl", "wb") as f:
    pickle.dump(model, f)

train = df_train.reset_index(drop=True)
valid = df_valid.reset_index(drop=True)
test = df_test.reset_index(drop=True)

X_train = torch.stack([trans(text) for text in df_train['TITLE']])
X_valid = torch.stack([trans(text) for text in df_valid['TITLE']])
X_test = torch.stack([trans(text) for text in df_test['TITLE']])

category = {'b':0,'t':1,'e':2,'m':3}
y_train = torch.tensor(df_train['CATEGORY'].map(lambda x:category[x]).values)
y_valid = torch.tensor(df_valid['CATEGORY'].map(lambda x:category[x]).values)
y_test = torch.tensor(df_test['CATEGORY'].map(lambda x:category[x]).values)

torch.save(X_train, 'X_train.pt')
torch.save(X_valid, 'X_valid.pt')
torch.save(X_test, 'X_test.pt')
torch.save(y_train, 'y_train.pt')
torch.save(y_valid, 'y_valid.pt')
torch.save(y_test, 'y_test.pt')