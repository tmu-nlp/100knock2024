#56.適合率，再現率，F1スコアの計測
"""
52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．
カテゴリごとに適合率，再現率，F1スコアを求め,
カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ．
"""

import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, classification_report

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

# 評価データ上での予測
test_predictions = model.predict(X_test)

# 適合率、再現率、F1スコアの計測
precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_predictions, average=None, labels=model.classes_)

# カテゴリごとの性能を表示
performance_df = pd.DataFrame({
    'Category': model.classes_,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})

print("Category-wise Performance:")
print(performance_df)

# マイクロ平均とマクロ平均の計算
micro_avg = precision_recall_fscore_support(y_test, test_predictions, average='micro')
macro_avg = precision_recall_fscore_support(y_test, test_predictions, average='macro')

print("\nMicro-average Performance:")
print(f"Precision: {micro_avg[0]:.4f}, Recall: {micro_avg[1]:.4f}, F1 Score: {micro_avg[2]:.4f}")

print("\nMacro-average Performance:")
print(f"Precision: {macro_avg[0]:.4f}, Recall: {macro_avg[1]:.4f}, F1 Score: {macro_avg[2]:.4f}")

# カテゴリごとの詳細なレポート
print("\nDetailed Classification Report:")
print(classification_report(y_test, test_predictions, target_names=model.classes_))

"""
Category-wise Performance:
  Category  Precision    Recall  F1 Score
0        b   0.884488  0.960573  0.920962
1        e   0.887395  0.975970  0.929577
2        m   0.951220  0.487500  0.644628
3        t   0.956522  0.567742  0.712551

Micro-average Performance:
Precision: 0.8928, Recall: 0.8928, F1 Score: 0.8928

Macro-average Performance:
Precision: 0.9199, Recall: 0.7479, F1 Score: 0.8019

Detailed Classification Report:
              precision    recall  f1-score   support

           b       0.88      0.96      0.92       558
           e       0.89      0.98      0.93       541
           m       0.95      0.49      0.64        80
           t       0.96      0.57      0.71       155

    accuracy                           0.89      1334
   macro avg       0.92      0.75      0.80      1334
weighted avg       0.90      0.89      0.88      1334
"""