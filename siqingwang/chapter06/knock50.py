# 50. data prepare

#bash
#unzip NewsAggregatorDataset.zip

import pandas as pd
from sklearn.model_selection import train_test_split

#ファイルを読み込む
data = pd.read_csv('newsCorpora.csv', sep = '\t', header = None, names = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

#事例（記事）を抽出する
publishers = ['Reuters', 'Huffington Post', 'Businessweek', '“Contactmusic.com', 'Daily Mail']
##isin:探す
data = data[data['PUBLISHER'].isin(publishers)]
data = data[['TITLE', 'CATEGORY']]

#分割する
##shuffle：分割する前dataをランダムにする
train, valid_test = train_test_split(data, test_size = 0.2, random_state = 0, shuffle = True, stratify = data['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, random_state=0, shuffle = True, stratify=valid_test['CATEGORY'])

train.to_csv('train.txt', sep = '\t', index = False, header=None)
valid.to_csv('valid.txt', sep = '\t', index = False, header=None)
test.to_csv('test.txt', sep = '\t', index = False, header=None)