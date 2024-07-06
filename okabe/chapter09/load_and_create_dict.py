import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict
import string
from torch.utils.data import Dataset
from knock80 import make_train_valid_test, make_word2id_dict
from knock81 import CreateDataset

# load data
df = pd.read_csv('newsCorpora.csv',
                 header=None,
                 sep='\t',
                 quoting=3,
                 names=['ID',
                        'TITLE',
                        'URL',
                        'PUBLISHER',
                        'CATEGORY',
                        'STORY',
                        'HOSTNAME',
                        'TIMESTAMP'])

df = df.loc[df['PUBLISHER'].isin(['Reuters',
                                  'Huffington Post',
                                  'Businessweek',
                                  'Contactmusic.com',
                                  'Daily Mail']),
                                  ['TITLE', 'CATEGORY']]
train, valid, test = make_train_valid_test(df)
word2id = make_word2id_dict(train['TITLE'])

#label
category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
y_train = train['CATEGORY'].map(lambda x: category_dict[x]).values
y_valid = valid['CATEGORY'].map(lambda x: category_dict[x]).values
y_test = test['CATEGORY'].map(lambda x: category_dict[x]).values

def tokenizer(text, word2id, unk=0):
  table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  return [word2id.get(word, unk) for word in text.translate(table).split()]

# Dataset
dataset_train = CreateDataset(train['TITLE'], y_train, tokenizer)
dataset_valid = CreateDataset(valid['TITLE'], y_valid, tokenizer)
dataset_test = CreateDataset(test['TITLE'], y_test, tokenizer)