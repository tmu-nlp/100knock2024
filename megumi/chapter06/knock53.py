#53.予測
#52で学習したロジスティック回帰モデルを用い，
# 与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．

import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 前処理（関数）
def preprocessing(text):
    text = "".join([i for i in text if i not in string.punctuation])
    text = text.lower()
    text = re.sub("[0-9]+", "", text)
    return text

# データの読み込み
train = pd.read_csv('train.txt', sep='\t', header=None, names=['CATEGORY', 'TITLE'])

# 前処理を行う
train['TITLE'] = train['TITLE'].map(preprocessing)

# TF-IDFベクトル化
vec_tfidf = TfidfVectorizer()
X_train = vec_tfidf.fit_transform(train['TITLE'])

# 目的変数
y_train = train['CATEGORY']

# ロジスティック回帰モデルの学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 予測関数の定義
def predict_category(title):
    # 前処理
    processed_title = preprocessing(title)
    # TF-IDFベクトル化
    title_vector = vec_tfidf.transform([processed_title])
    # カテゴリ予測
    category = model.predict(title_vector)[0]
    # 予測確率
    probabilities = model.predict_proba(title_vector)[0]
    # カテゴリと確率のペアを作成
    categories = model.classes_
    category_probabilities = {cat: prob for cat, prob in zip(categories, probabilities)}
    return category, category_probabilities

# テスト用の記事見出し
test_title = "Apple unveils new iPhone model"
predicted_category, predicted_probabilities = predict_category(test_title)

print("Predicted Category:", predicted_category)
print("Predicted Probabilities:", predicted_probabilities)

"""
Predicted Category: t
Predicted Probabilities: {'b': 0.021408443050618012, 'e': 0.024067449397960853, 'm': 0.007761371281128917, 't': 0.9467627362702922}
"""