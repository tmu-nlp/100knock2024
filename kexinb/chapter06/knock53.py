# task53. 予測
# 52で学習したロジスティック回帰モデルを用い，
# 与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．

import pickle
import numpy as np
import pandas as pd

def score_lr(lr, x): # -> [max.prob, pred.label]
    return [np.max(lr.predict_proba(x), axis=1), lr.predict(x)]

# load model
lr = pickle.load(open("output/ch6/logreg.pkl", 'rb'))

X_train = pd.read_table("output/ch6/train.feature.txt")
X_test = pd.read_table("output/ch6/test.feature.txt")

train_pred = score_lr(lr, X_train)
test_pred = score_lr(lr, X_test)

if __name__ == "__main__":
    print(train_pred)

# [array([0.91230316, 0.4089398 , 0.65182876, ..., 0.88564153, 0.9431732 ,
#    0.90615187]), array(['b', 'e', 'e', ..., 'e', 'b', 'b'], dtype=object)]