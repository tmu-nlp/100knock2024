import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 特徴量データの読み込み
X_train = joblib.load('train.feature.pkl')
X_valid = joblib.load('valid.feature.pkl')
X_test = joblib.load('test.feature.pkl')

# ラベルデータの読み込み
train_data = pd.read_csv('train.txt', sep='\t', header=None)
y_train = train_data[0]

valid_data = pd.read_csv('valid.txt', sep='\t', header=None)
y_valid = valid_data[0]

test_data = pd.read_csv('test.txt', sep='\t', header=None)
y_test = test_data[0]
# 正則化パラメータのリスト
C_values = [0.01, 0.1, 1, 10, 100]

# 正解率を保持するリスト
train_accuracies = []
valid_accuracies = []
test_accuracies = []

# 各正則化パラメータでモデルを学習し、正解率を計測
for C in C_values:
    clf = LogisticRegression(penalty='l2', C=C, solver='sag', random_state=0, max_iter=10000)
    clf.fit(X_train, y_train)
    
    # 学習データの正解率
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracies.append(train_accuracy)
    
    # 検証データの正解率
    y_valid_pred = clf.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    valid_accuracies.append(valid_accuracy)
    
    # 評価データの正解率
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_accuracy)

# 結果の表示
print("Train Accuracies:", train_accuracies)
print("Validation Accuracies:", valid_accuracies)
print("Test Accuracies:", test_accuracies)
# グラフのプロット
plt.figure(figsize=(10, 6))
plt.plot(C_values, train_accuracies, marker='o', label='Train Accuracy')
plt.plot(C_values, valid_accuracies, marker='o', label='Validation Accuracy')
plt.plot(C_values, test_accuracies, marker='o', label='Test Accuracy')

# グラフの設定
plt.xscale('log')
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Accuracy')
plt.title('Effect of Regularization Parameter on Accuracy')
plt.legend()
plt.grid(True)
plt.show()
