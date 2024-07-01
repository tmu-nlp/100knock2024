'''
80. ID番号への変換Permalink
問題51で構築した学習データ中の単語にユニークなID番号を付与したい．
学習データ中で最も頻出する単語に1，2番目に頻出する単語に2，……といった方法で，
学習データ中で2回以上出現する単語にID番号を付与せよ．そして，与えられた単語列に対して，
ID番号の列を返す関数を実装せよ．
ただし，出現頻度が2回未満の単語のID番号はすべて0とせよ．
'''

from collections import defaultdict
import string
import pandas as pd
from sklearn.model_selection import train_test_split

def make_train_valid_test(df):
    train, valid_test = train_test_split(df, test_size=0.2,
                                         shuffle=True,
                                         random_state=123,
                                         stratify=df['CATEGORY'])

    valid, test = train_test_split(valid_test,
                                   test_size=0.5,
                                   shuffle=True,
                                   random_state=123,
                                   stratify=valid_test['CATEGORY'])

    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return train, valid, test

def make_word2id_dict(text_list):
    d = defaultdict(int)
    #句読点をスペースに変換するテーブル
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    for text in text_list:
        for word in text.translate(table).split():
            d[word] += 1
    d = sorted(d.items(), key=lambda x:x[1], reverse=True)

    #出現2回以上にid
    word2id = {word: i + 1 for i, (word, cnt) in enumerate(d) if cnt > 1}
    return word2id

def tokenizer(text, word2id, unk=0):
  table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  return [word2id.get(word, unk) for word in text.translate(table).split()]


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

# test
text = train.iloc[1, train.columns.get_loc('TITLE')]
id_list = tokenizer(text, word2id)

print(f'テキスト: {text}')
print(f'ID列: {id_list}')

"""
output:
テキスト: Amazon Plans to Fight FTC Over Mobile-App Purchases
ID列: [169, 539, 1, 683, 1237, 82, 279, 1898, 4199]
"""