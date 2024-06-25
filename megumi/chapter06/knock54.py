#54. 正解率の計測
#52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ．

import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 前処理（関数）
def preprocessing(text):
    text = "".join([i for i in text if i not in string.punctuation])
    text = text.lower()
    text = re.sub("[0-9]+", "", text)
    return text

# データの読み込み
train = pd.read_csv('train.txt', sep='\t', header=None, names=['CATEGORY', 'TITLE'])
test = pd.read_csv('test.txt', sep='\t', header=None, names=['CATEGORY', 'TITLE'])

# 前処理を行う
train['TITLE'] = train['TITLE'].map(preprocessing)
test['TITLE'] = test['TITLE'].map(preprocessing)

# TF-IDFベクトル化
vec_tfidf = TfidfVectorizer()
X_train = vec_tfidf.fit_transform(train['TITLE'])
X_test = vec_tfidf.transform(test['TITLE'])

# 目的変数
y_train = train['CATEGORY']
y_test = test['CATEGORY']

# ロジスティック回帰モデルの学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 学習データ上での正解率
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Train Accuracy: {train_accuracy}")

# 評価データ上での正解率
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy}")

"""
Train Accuracy: 0.9433095952023988
Test Accuracy: 0.8928035982008995
"""
