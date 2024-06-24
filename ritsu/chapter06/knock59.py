from knock52 import load_features, load_labels
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import lightgbm as lgb

def train_model(features, labels, model_type, params):
    if model_type == 'LogisticRegression':
        model = LogisticRegression(**params, max_iter=1000)
    elif model_type == 'SVM':
        model = SVC(**params)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(**params)
    elif model_type == 'LightGBM':
        train_data = lgb.Dataset(features, label=labels)
        model = lgb.train(params, train_data)
    elif model_type == 'MLP':
        model = MLPClassifier(**params, max_iter=1000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type != 'LightGBM':
        model.fit(features, labels)

    return model

def evaluate_model(model, features, labels, model_type):
    if model_type == 'LightGBM':
        predictions = model.predict(features)
    else:
        predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    return accuracy

def main():
    # 特徴量の読み込み
    train_features = load_features("train.feature.txt")
    valid_features = load_features("valid.feature.txt")
    test_features = load_features("test.feature.txt")

    # ラベルの読み込み
    train_labels = load_labels("train.txt")
    valid_labels = load_labels("valid.txt")
    test_labels = load_labels("test.txt")

    # ハイパーパラメータの設定
    hyperparameters = {
        'RandomForest': [
            {'n_estimators': 10, 'max_depth': 2},
            {'n_estimators': 10, 'max_depth': 4},
            {'n_estimators': 10, 'max_depth': 6},
            {'n_estimators': 50, 'max_depth': 2},
            {'n_estimators': 50, 'max_depth': 4},
            {'n_estimators': 50, 'max_depth': 6},
            {'n_estimators': 100, 'max_depth': 2},
            {'n_estimators': 100, 'max_depth': 4},
            {'n_estimators': 100, 'max_depth': 6}
        ],
        'LightGBM': [
            {'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.1, 'n_estimators': 50},
            {'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.1, 'n_estimators': 100},
            {'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.1, 'n_estimators': 200},
            {'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.01, 'n_estimators': 50},
            {'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.01, 'n_estimators': 100},
            {'boosting_type': 'gbdt', 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.01, 'n_estimators': 200}
        ],
        'MLP': [
            {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam'},
            {'hidden_layer_sizes': (200,), 'activation': 'relu', 'solver': 'adam'},
            {'hidden_layer_sizes': (100, 100), 'activation': 'relu', 'solver': 'adam'},
            {'hidden_layer_sizes': (200, 200), 'activation': 'relu', 'solver': 'adam'},
            {'hidden_layer_sizes': (100,), 'activation': 'tanh', 'solver': 'adam'},
            {'hidden_layer_sizes': (200,), 'activation': 'tanh', 'solver': 'adam'},
            {'hidden_layer_sizes': (100, 100), 'activation': 'tanh', 'solver': 'adam'},
            {'hidden_layer_sizes': (200, 200), 'activation': 'tanh', 'solver': 'adam'}
        ]
    }

    results = []

    # ハイパーパラメータを変えながらモデルを学習し、正解率を評価
    for model_type, params_list in hyperparameters.items():
        for params in params_list:
            model = train_model(train_features, train_labels, model_type, params)
            valid_accuracy = evaluate_model(model, valid_features, valid_labels, model_type)
            test_accuracy = evaluate_model(model, test_features, test_labels, model_type)
            results.append({
                'Model': model_type,
                'Params': params,
                'Valid Accuracy': valid_accuracy,
                'Test Accuracy': test_accuracy
            })

    # 結果をデータフレームに変換
    df_results = pd.DataFrame(results)

    # 検証データ上の正解率が最も高いモデルとパラメータを取得
    best_model = df_results.loc[df_results['Valid Accuracy'].idxmax()]

    print("Best Model:")
    print(f"Model: {best_model['Model']}")
    print(f"Params: {best_model['Params']}")
    print(f"Valid Accuracy: {best_model['Valid Accuracy']:.3f}")
    print(f"Test Accuracy: {best_model['Test Accuracy']:.3f}")

if __name__ == "__main__":
    main()