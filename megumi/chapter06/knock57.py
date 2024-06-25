#57. 特徴量の重みの確認
#52で学習したロジスティック回帰モデルの中で，重みの高い特徴量トップ10と，重みの低い特徴量トップ10を確認せよ．

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

# 特徴量の重みを確認
feature_names = vec_tfidf.get_feature_names_out()
coefficients = model.coef_[0]

# 重みの高い特徴量トップ10
top10_positive_coefficients = coefficients.argsort()[-10:][::-1]
top10_positive_features = [(feature_names[i], coefficients[i]) for i in top10_positive_coefficients]

# 重みの低い特徴量トップ10
top10_negative_coefficients = coefficients.argsort()[:10]
top10_negative_features = [(feature_names[i], coefficients[i]) for i in top10_negative_coefficients]

print("重みの高い特徴量トップ10:")
for feature, weight in top10_positive_features:
    print(f"{feature}: {weight}")

print("\n重みの低い特徴量トップ10:")
for feature, weight in top10_negative_features:
    print(f"{feature}: {weight}")
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

# 特徴量の重みを確認
feature_names = vec_tfidf.get_feature_names_out()
coefficients = model.coef_[0]

# 重みの高い特徴量トップ10
top10_positive_coefficients = coefficients.argsort()[-10:][::-1]
top10_positive_features = [(feature_names[i], coefficients[i]) for i in top10_positive_coefficients]

# 重みの低い特徴量トップ10
top10_negative_coefficients = coefficients.argsort()[:10]
top10_negative_features = [(feature_names[i], coefficients[i]) for i in top10_negative_coefficients]

print("重みの高い特徴量トップ10:")
for feature, weight in top10_positive_features:
    print(f"{feature}: {weight}")

print("\n重みの低い特徴量トップ10:")
for feature, weight in top10_negative_features:
    print(f"{feature}: {weight}")

"""
重みの高い特徴量トップ10:
bank: 3.386047068679511
fed: 3.155911696333121
china: 3.0210876196540073
ecb: 2.844210330169248
euro: 2.6027882785734313
ukraine: 2.580636522579256
stocks: 2.5667536661790837
update: 2.531273336267991
oil: 2.4861068245426567
profit: 2.3931474446180627

重みの低い特徴量トップ10:
and: -2.4340951336900285
the: -2.161097410297653
her: -2.0213813459114593
ebola: -1.9193848562397862
she: -1.6955987074594685
video: -1.6893196890395064
apple: -1.6434506899760593
study: -1.6214794971292903
kardashian: -1.610194015451075
fda: -1.5931592864512027
重みの高い特徴量トップ10:
bank: 3.386047068679511
fed: 3.155911696333121
china: 3.0210876196540073
ecb: 2.844210330169248
euro: 2.6027882785734313
ukraine: 2.580636522579256
stocks: 2.5667536661790837
update: 2.531273336267991
oil: 2.4861068245426567
profit: 2.3931474446180627

重みの低い特徴量トップ10:
and: -2.4340951336900285
the: -2.161097410297653
her: -2.0213813459114593
ebola: -1.9193848562397862
she: -1.6955987074594685
video: -1.6893196890395064
apple: -1.6434506899760593
study: -1.6214794971292903
kardashian: -1.610194015451075
fda: -1.5931592864512027
"""