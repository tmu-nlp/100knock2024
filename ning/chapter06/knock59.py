import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# アルゴリズムとパラメータの設定
models = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=10000, solver='saga', random_state=0),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=0),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=0),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None]
        }
    }
}
best_models = {}

for model_name, model_info in models.items():
    clf = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    
    best_models[model_name] = {
        'best_params': clf.best_params_,
        'best_score': clf.best_score_,
        'best_estimator': clf.best_estimator_
    }

# 検証データ上での評価
for model_name, model_info in best_models.items():
    y_valid_pred = model_info['best_estimator'].predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    model_info['valid_accuracy'] = valid_accuracy
    print(f"{model_name} - Best Params: {model_info['best_params']}, Validation Accuracy: {valid_accuracy:.4f}")
# 検証データ上で最も高い正解率を示したモデルを選択
best_model_name = max(best_models, key=lambda name: best_models[name]['valid_accuracy'])
best_model = best_models[best_model_name]['best_estimator']

# 評価データ上での正解率を計測
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Best Model: {best_model_name}")
print(f"Test Accuracy: {test_accuracy:.4f}")
