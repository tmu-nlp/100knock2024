"""
1.データの読み込み
2.特徴量とラベルの分割
  Category 列をラベル (y_train) とし、それ以外の列を特徴量 (X_train) として分割
3.ロジスティック回帰モデルの学習
　　scikit-learn の LogisticRegression クラスを用いてモデルを作成し、学習データ (X_train, y_train) を使う
4.学習済みモデルをjobilbで保存
"""
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# データの読み込み
train_df = pd.read_csv('train.feature.txt', sep='\t')

# 特徴量とラベルの分割
X_train = train_df.drop(columns=['Category', 'Title'])
y_train = train_df['Category']

# ロジスティック回帰モデルの学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 学習済みモデルの保存
joblib.dump(model, 'logistic_regression_model.pkl')
