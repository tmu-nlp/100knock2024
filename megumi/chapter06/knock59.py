#59. ハイパーパラメータの探索
"""
学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．
検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．
また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．
"""
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# 前処理（関数）
def preprocessing(text):
    text = "".join([i for i in text if i not in string.punctuation])
    text = text.lower()
    text = re.sub("[0-9]+", "", text)
    return text

# データの読み込み
train = pd.read_csv('train.txt', sep='\t', header=None, names=['CATEGORY', 'TITLE'])
test = pd.read_csv('test.txt', sep='\t', header=None, names=['CATEGORY', 'TITLE'])

# 前処理を行う
train['TITLE'] = train['TITLE'].map(preprocessing)
test['TITLE'] = test['TITLE'].map(preprocessing)

# 学習データを学習データと検証データに分割
train_data, val_data, train_labels, val_labels = train_test_split(
    train['TITLE'], train['CATEGORY'], test_size=0.2, random_state=42
)

# TF-IDFベクトル化
vec_tfidf = TfidfVectorizer()
X_train = vec_tfidf.fit_transform(train_data)
X_val = vec_tfidf.transform(val_data)
X_test = vec_tfidf.transform(test['TITLE'])

# 目的変数
y_train = train_labels
y_val = val_labels
y_test = test['CATEGORY']

# モデルとパラメータの設定
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVC': SVC(),
    'RandomForest': RandomForestClassifier()
}

params = {
    'LogisticRegression': {'C': [0.01, 0.1, 1, 10, 100]},
    'SVC': {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']},
    'RandomForest': {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30]}
}

# 最適なモデルとパラメータの探索
best_score = 0
best_model = None
best_params = None

for model_name in models:
    model = models[model_name]
    param = params[model_name]
    grid_search = GridSearchCV(model, param, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

# 最適なモデルで評価データ上の正解率を計測
best_model.fit(X_train, y_train)
test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"Best Model: {best_model}")
print(f"Best Parameters: {best_params}")
print(f"Test Accuracy: {test_accuracy}")

"""
Best Model: SVC(C=10, kernel='linear')
Best Parameters: {'C': 10, 'kernel': 'linear'}
Test Accuracy: 0.9062968515742129
"""