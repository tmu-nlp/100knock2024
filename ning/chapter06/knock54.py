import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# データの読み込み
train_df = pd.read_csv('train.feature.txt', sep='\t')
valid_df = pd.read_csv('valid.feature.txt', sep='\t')

# 特徴量とラベルの分割
X_train = train_df.drop(columns=['Category', 'Title'])
y_train = train_df['Category']

X_valid = valid_df.drop(columns=['Category', 'Title'])
y_valid = valid_df['Category']

# 学習済みモデルの読み込み
model = joblib.load('logistic_regression_model.pkl')

# 学習データ上での正解率計測
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f'Training Accuracy: {train_accuracy:.4f}')

# 評価データ上での正解率計測
y_valid_pred = model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print(f'Validation Accuracy: {valid_accuracy:.4f}')

# Training Accuracy: 0.9898
# Validation Accuracy: 0.8988