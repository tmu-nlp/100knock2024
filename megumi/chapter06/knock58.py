#58. 正則化パラメータの変更
"""
ロジスティック回帰モデルを学習するとき，正則化パラメータを調整することで，学習時の過学習（overfitting）の度合いを制御できる．
異なる正則化パラメータでロジスティック回帰モデルを学習し，学習データ，検証データ，および評価データ上の正解率を求めよ．
実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．
"""

import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

# 学習データを学習データと検証データに分割
train_data, val_data, train_labels, val_labels = train_test_split(
    train['TITLE'], train['CATEGORY'], test_size=0.2, random_state=42
)

# TF-IDFベクトル化
vec_tfidf = TfidfVectorizer()
X_train = vec_tfidf.fit_transform(train_data)
X_val = vec_tfidf.transform(val_data)
X_test = vec_tfidf.transform(test['TITLE'])

# 目的変数
y_train = train_labels
y_val = val_labels
y_test = test['CATEGORY']

# 正則化パラメータのリスト
C_values = [0.01, 0.1, 1, 10, 100]

# 正解率を格納するリスト
train_accuracies = []
val_accuracies = []
test_accuracies = []

# 異なる正則化パラメータでモデルを学習し、正解率を計測
for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    train_accuracies.append(accuracy_score(y_train, train_pred))
    val_accuracies.append(accuracy_score(y_val, val_pred))
    test_accuracies.append(accuracy_score(y_test, test_pred))

# グラフの作成
plt.figure(figsize=(10, 6))
plt.plot(C_values, train_accuracies, marker='o', label='Train Accuracy')
plt.plot(C_values, val_accuracies, marker='o', label='Validation Accuracy')
plt.plot(C_values, test_accuracies, marker='o', label='Test Accuracy')
plt.xscale('log')
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Accuracy')
plt.title('Effect of Regularization Parameter on Accuracy')
plt.legend()
plt.grid(True)
plt.show()
