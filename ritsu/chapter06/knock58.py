from knock52 import load_features, load_labels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train_model(features, labels, C):
    model = LogisticRegression(C=C, max_iter=1000)
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

    # 正則化パラメータの設定
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    # 正解率を格納するリスト
    train_accuracies = []
    valid_accuracies = []
    test_accuracies = []

    # 正則化パラメータを変えながらモデルを学習し、正解率を評価
    for C in C_values:
        model = train_model(train_features, train_labels, C)
        train_accuracy = evaluate_model(model, train_features, train_labels)
        valid_accuracy = evaluate_model(model, valid_features, valid_labels)
        test_accuracy = evaluate_model(model, test_features, test_labels)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)
        test_accuracies.append(test_accuracy)

    # グラフのプロット
    plt.plot(C_values, train_accuracies, marker='o', label='Train')
    plt.plot(C_values, valid_accuracies, marker='o', label='Valid')
    plt.plot(C_values, test_accuracies, marker='o', label='Test')
    plt.xscale('log')
    plt.xlabel('Regularization Parameter (C)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('knock58.png')
    plt.show()

if __name__ == "__main__":
    main()