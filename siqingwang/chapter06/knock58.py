import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np


# Preprocess the data

# LabelEncoder(): function from sklearn, カテゴリ変数を数字に変化させる関数
# TOKENSから抽出した特徴がありますからここでもうTOKENSを使わないんです
def preprocess_features(features):
    le = LabelEncoder()
    features['CATEGORY'] = le.fit_transform(features['CATEGORY'])
    X = features.drop(columns=['CATEGORY', 'TOKENS'])
    y = features['CATEGORY']
    return X, y, le

# ファイルの読み込み
train_features = pd.read_csv('train.feature.txt', sep='\t')
valid_features = pd.read_csv('valid.feature.txt', sep='\t')
test_features = pd.read_csv('test.feature.txt', sep='\t')

X_train, y_train, label_encoder = preprocess_features(train_features)
X_valid, y_valid, _ = preprocess_features(valid_features)
X_test, y_test, _ = preprocess_features(test_features)

# Define a range of regularization parameters
# 正則パラメータのリストに
regularization_params = [0.01, 0.1, 1, 10, 100]

# Store the accuracy results
# accuracy結果を保存するために空のリストを作ります
train_accuracies = []
valid_accuracies = []
test_accuracies = []

# 設定された正則化のパラメータを、一つずつ試して、logitモデルに組み込んで
for C in regularization_params:
    # Initialize the logistic regression model with the given regularization parameter
    model = LogisticRegression(C=C, max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    valid_accuracy = accuracy_score(y_valid, model.predict(X_valid))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    # Store the results
    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)
    test_accuracies.append(test_accuracy)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(regularization_params, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(regularization_params, valid_accuracies, label='Validation Accuracy', marker='o')
plt.plot(regularization_params, test_accuracies, label='Test Accuracy', marker='o')
plt.xscale('log')
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Accuracy Score')
plt.title('Effect of Regularization on Model Accuracy')
plt.legend()
plt.grid(True)
plt.show()
