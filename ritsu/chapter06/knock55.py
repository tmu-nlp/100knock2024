from knock52 import load_features, load_labels, train_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(labels, predictions, label_names, filename):
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=label_names, yticklabels=label_names, square=True, ax=ax)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)  # 混同行列を画像ファイルとして保存
    plt.close()

def main():
    # 特徴量の読み込み
    train_features = load_features("train.feature.txt")
    test_features = load_features("test.feature.txt")

    # ラベルの読み込み
    train_labels = load_labels("train.txt")
    test_labels = load_labels("test.txt")

    # モデルの学習
    model = train_model(train_features, train_labels)

    # 学習データの予測
    train_predictions = model.predict(train_features)

    # 評価データの予測
    test_predictions = model.predict(test_features)

    # ラベル名の定義
    label_names = ['b', 'e', 't', 'm']

    # 学習データの混同行列のプロットと保存
    print("学習データの混同行列:")
    plot_confusion_matrix(train_labels, train_predictions, label_names, "knock55_train.png")

    # 評価データの混同行列のプロットと保存
    print("評価データの混同行列:")
    plot_confusion_matrix(test_labels, test_predictions, label_names, "knock55_test.png")

if __name__ == "__main__":
    main()