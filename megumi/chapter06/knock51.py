#51.特徴量抽出
"""
学習データ，検証データ，評価データから特徴量を抽出し，
それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． 
なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．
"""
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer

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
train_features = vec_tfidf.fit_transform(train['TITLE'])
valid_features = vec_tfidf.transform(valid['TITLE'])
test_features = vec_tfidf.transform(test['TITLE'])

# データフレームに変換
train_features_df = pd.DataFrame(train_features.toarray(), columns=vec_tfidf.get_feature_names_out())
valid_features_df = pd.DataFrame(valid_features.toarray(), columns=vec_tfidf.get_feature_names_out())
test_features_df = pd.DataFrame(test_features.toarray(), columns=vec_tfidf.get_feature_names_out())

# 特徴量をファイルに保存
train_features_df.to_csv('train.feature.txt', sep='\t', index=False)
valid_features_df.to_csv('valid.feature.txt', sep='\t', index=False)
test_features_df.to_csv('test.feature.txt', sep='\t', index=False)
