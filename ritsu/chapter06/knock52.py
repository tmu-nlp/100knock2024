import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_features(filepath):
    return pd.read_csv(filepath, sep="\t", header=None)

def load_labels(filepath):
    return pd.read_csv(filepath, sep="\t", header=None, usecols=[0])

def train_model(features, labels):
    model = LogisticRegression(max_iter=1000)
    model.fit(features, labels)
    return model

def evaluate_model(model, features, labels):
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

    # モデルの学習
    model = train_model(train_features, train_labels)

    # モデルの評価
    train_accuracy = evaluate_model(model, train_features, train_labels)
    valid_accuracy = evaluate_model(model, valid_features, valid_labels)
    test_accuracy = evaluate_model(model, test_features, test_labels)

    print(f"訓練データの正解率: {train_accuracy:.3f}")
    print(f"検証データの正解率: {valid_accuracy:.3f}")
    print(f"テストデータの正解率: {test_accuracy:.3f}")

if __name__ == "__main__":
    main()