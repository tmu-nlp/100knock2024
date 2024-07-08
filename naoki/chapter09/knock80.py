#80
from collections import defaultdict
import joblib
import pandas as pd
import numpy as np

#ファイルの読み込み
header_name = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
X_train = pd.read_table('drive/MyDrive/df_train.txt', header=None, names=header_name)

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

word2token = dict(zip(words, idx))
print(word2token)