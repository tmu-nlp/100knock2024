from google.colab import drive
drive.mount('/content/drive/')

# 50. data prepare

#bash
#unzip NewsAggregatorDataset.zip

import pandas as pd
from sklearn.model_selection import train_test_split

#ファイルを読み込む
data = pd.read_csv('/content/drive/My Drive/NLP/news+aggregator/newsCorpora.csv', sep = '\t', header = None, names = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

#事例（記事）を抽出する
publishers = ['Reuters', 'Huffington Post', 'Businessweek', '“Contactmusic.com', 'Daily Mail']
##isin:探す
data = data[data['PUBLISHER'].isin(publishers)]
data = data[['TITLE', 'CATEGORY']]

#分割する
##shuffle：分割する前dataをランダムにする
train, valid_test = train_test_split(data, test_size = 0.2, random_state = 0, shuffle = True, stratify = data['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, random_state=0, shuffle = True, stratify=valid_test['CATEGORY'])

#knock60の単語ベクトル
from gensim.models import KeyedVectors
file = '/content/drive/MyDrive/NLP/GoogleNews-vectors-negative300.bin.gz'
model = KeyedVectors.load_word2vec_format(file, binary = True)

# 70

import string
import torch


def tensor1(text):
  table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  words = text.translate(table).split()
  vec = [model[word] for word in words if word in model]  # 1語ずつベクトル化
  return torch.tensor(sum(vec)/len(vec))


X_train = torch.stack([tensor1(text) for text in train['TITLE']])
X_valid = torch.stack([tensor1(text) for text in valid['TITLE']])
X_test = torch.stack([tensor1(text) for text in test['TITLE']])

category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = torch.tensor(train['CATEGORY'].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test['CATEGORY'].map(lambda x: category_dict[x]).values)

torch.save(X_train, '/content/drive/MyDrive/NLP/X_train.pt')
torch.save(X_valid, '/content/drive/MyDrive/NLP/X_valid.pt')
torch.save(X_test, '/content/drive/MyDrive/NLP/X_test.pt')
torch.save(y_train, '/content/drive/MyDrive/NLP/y_train.pt')
torch.save(y_valid, '/content/drive/MyDrive/NLP/y_valid.pt')
torch.save(y_test, '/content/drive/MyDrive/NLP/y_test.pt')