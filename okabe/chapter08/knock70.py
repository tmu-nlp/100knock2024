'''
70. 単語ベクトルの和による特徴量
問題50で構築した学習データ，検証データ，評価データを行列・ベクトルに変換したい．
例えば，学習データについて，すべての事例xiの特徴ベクトルxiを並べた行列Xと，
正解ラベルを並べた行列（ベクトル）Yを作成したい．
'''
import pandas as pd
from gensim.models import KeyedVectors
import numpy as np


train = pd.read_csv("train.txt", sep="\t")
valid = pd.read_csv("valid.txt", sep="\t")
test = pd.read_csv("test.txt", sep="\t")


model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300_ch08.bin.gz", binary=True)

#category name into discrete variable
labels = {"b":0, "t":1, "e":2, "m":3}
y_train = train.iloc[:,1].replace(labels)
y_valid = valid.iloc[:,1].replace(labels)
y_test = test.iloc[:,1].replace(labels)

y_train = np.delete(y_train,0,0)

y_train.to_csv("./data/y_train.txt", header=False, index=False)
y_valid.to_csv("./data/y_valid.txt", header=False, index=False)
y_test.to_csv("./data/y_test.txt", header=False, index=False)

#feature
def write_x(f_name, df):
    with open(f_name, "w") as f:
        for title in df.iloc[1:,0]:
            vectors = []
            for word in title.split():
                if word in model.index_to_key:
                    vectors.append(model[word])
            if len(vectors) == 0:
                vector = np.zeros(300) #if empty
            else:
                vectors = np.array(vectors)
                vector = vectors.mean(axis=0)
            vector = vector.astype("str").tolist()
            output = " ".join(vector)
            print(output, file=f)

write_x("./data/x_train.txt", train)
write_x("./data/x_valid.txt", valid)
write_x("./data/x_test.txt", test)