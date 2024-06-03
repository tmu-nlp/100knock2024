"""
53. 
52で学習したロジスティック回帰モデルを用い、
与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．
"""

import numpy as np
from knock51 import X_train, X_test, X_valid
from knock52 import lr


def pred_and_prob_lr(lr, X):
    '''第一要素に予測されるラベルのリストとするリスト、 第二要素を予測確率のndarrayを返す'''
    return [lr.predict(X),
            np.max(lr.predict_proba(X), axis=1)]


train_pred = pred_and_prob_lr(lr, X_train)
test_pred = pred_and_prob_lr(lr, X_test)
valid_pred = pred_and_prob_lr(lr, X_valid)

if __name__ == '__main__':
    print(f'train_pred score:{train_pred}')