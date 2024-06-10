# task59. ハイパーパラメータの探索
# 学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．
# 検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．
# また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv("output/ch6/train.feature.txt", sep='\t')
X_valid = pd.read_csv("output/ch6/valid.feature.txt", sep='\t')
X_test = pd.read_csv("output/ch6/test.feature.txt", sep='\t')

Y_train = pd.read_csv("output/ch6/train.txt", sep='\t', header=None, 
                      names=['TITLE', 'CATEGORY'])['CATEGORY']
Y_valid = pd.read_csv("output/ch6/valid.txt", sep='\t', header=None, 
                      names=['TITLE', 'CATEGORY'])['CATEGORY']
Y_test = pd.read_csv("output/ch6/test.txt", sep='\t', header=None, 
                      names=['TITLE', 'CATEGORY'])['CATEGORY']

param_grid = {'C': [i for i in range(1, 21)]}

grid_search = GridSearchCV(LogisticRegression(
    random_state=42, max_iter=1000), param_grid, cv=5)
grid_search.fit(X_train, Y_train)

print("Best parameters : {}".format(grid_search.best_params_))
print("Best cross-validation score : {:.3f}".format(grid_search.best_score_))