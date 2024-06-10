# task51. 特徴量抽出
'''
学習データ, 検証データ, 評価データから特徴量を抽出し, 
それぞれtrain.feature.txt, valid.feature.txt, test.feature.txtというファイル名で保存せよ.
なお, カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．
'''

import pandas as pd
import string

from sklearn.feature_extraction.text import CountVectorizer # BoW
from sklearn.feature_extraction.text import TfidfVectorizer # tf-idf

def preprocess(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = text.lower()
    text = ''.join([i for i in text if not i.isdigit()])
    return text

# Load and preprocess data
header_name = ['TITLE', 'CATEGORY']
train = pd.read_csv('output/ch6/train.txt', header=None, sep='\t', names=header_name)
valid = pd.read_csv('output/ch6/valid.txt', header=None, sep='\t', names=header_name)
test = pd.read_csv('output/ch6/test.txt', header=None, sep='\t', names=header_name)

# Concatenate data for preprocessing
df = pd.concat([train, valid, test], axis=0).reset_index(drop=True)
df['TITLE'] = df['TITLE'].apply(preprocess)

# Split back the data
train_valid_d = df[:len(train) + len(valid)]
test_d = df[len(train) + len(valid):]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(min_df=10, ngram_range=(1, 2))

# Fit and transform data
train_valid_f = vectorizer.fit_transform(train_valid_d["TITLE"])
test_f = vectorizer.transform(test_d["TITLE"])

# Convert to DataFrame and save with headers
train_valid_vec = pd.DataFrame(train_valid_f.toarray(), columns=vectorizer.get_feature_names_out())
test_vec = pd.DataFrame(test_f.toarray(), columns=vectorizer.get_feature_names_out())

train_vec = train_valid_vec[:len(train)]
valid_vec = train_valid_vec[len(train):]

train_vec.to_csv("output/ch6/train.feature.txt", sep="\t", index=False)
valid_vec.to_csv("output/ch6/valid.feature.txt", sep="\t", index=False)
test_vec.to_csv("output/ch6/test.feature.txt", sep="\t", index=False)