from knock52 import load_features, load_labels, train_model, evaluate_model

def main():
    # 特徴量の読み込み
    train_features = load_features("train.feature.txt")
    test_features = load_features("test.feature.txt")

    # ラベルの読み込み
    train_labels = load_labels("train.txt")
    test_labels = load_labels("test.txt")

    # モデルの学習
    model = train_model(train_features, train_labels)

    # モデルの評価
    train_accuracy = evaluate_model(model, train_features, train_labels)
    test_accuracy = evaluate_model(model, test_features, test_labels)

    print(f"学習データの正解率: {train_accuracy:.3f}")
    print(f"評価データの正解率: {test_accuracy:.3f}")

if __name__ == "__main__":
    main()