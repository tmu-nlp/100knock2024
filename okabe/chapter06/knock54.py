"""
54. 正解率の計測
52で学習したロジスティック回帰モデルの正解率を，
学習データおよび評価データ上で計測せよ
"""

from knock51 import X_train, X_test, train, test
from knock52 import lr

if __name__ == "__main__":
    print('train data score：', lr.score(X_train, train['CATEGORY']))
    print('test data score：', lr.score(X_test, test['CATEGORY']))