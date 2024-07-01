#52.学習
#51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．

import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 前処理（関数）
def preprocessing(text):
    text = "".join([i for i in text if i not in string.punctuation])
    text = text.lower()
    text = re.sub("[0-9]+", "", text)
    return text

# データの読み込み
train = pd.read_csv('train.txt', sep='\t', header=None, names=['CATEGORY', 'TITLE'])
valid = pd.read_csv('valid.txt', sep='\t', header=None, names=['CATEGORY', 'TITLE'])
test = pd.read_csv('test.txt', sep='\t', header=None, names=['CATEGORY', 'TITLE'])

# 前処理を行う
train['TITLE'] = train['TITLE'].map(preprocessing)
valid['TITLE'] = valid['TITLE'].map(preprocessing)
test['TITLE'] = test['TITLE'].map(preprocessing)

# TF-IDFベクトル化
vec_tfidf = TfidfVectorizer()
X_train = vec_tfidf.fit_transform(train['TITLE'])
X_valid = vec_tfidf.transform(valid['TITLE'])
X_test = vec_tfidf.transform(test['TITLE'])

# 目的変数
y_train = train['CATEGORY']
y_valid = valid['CATEGORY']
y_test = test['CATEGORY']

# ロジスティック回帰モデルの学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 検証データでの評価
y_valid_pred = model.predict(X_valid)
print("Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))
print("Validation Classification Report:\n", classification_report(y_valid, y_valid_pred))

# テストデータでの評価
y_test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

"""
               precision    recall  f1-score   support

           b       0.87      0.96      0.92       539
           e       0.89      0.97      0.93       560
           m       0.95      0.62      0.75        91
           t       0.84      0.44      0.58       144

    accuracy                           0.88      1334
   macro avg       0.89      0.75      0.79      1334
weighted avg       0.88      0.88      0.87      1334

Test Accuracy: 0.8928035982008995
Test Classification Report:
               precision    recall  f1-score   support

           b       0.88      0.96      0.92       558
           e       0.89      0.98      0.93       541
           m       0.95      0.49      0.64        80
           t       0.96      0.57      0.71       155

    accuracy                           0.89      1334
   macro avg       0.92      0.75      0.80      1334
weighted avg       0.90      0.89      0.88      1334

"""
