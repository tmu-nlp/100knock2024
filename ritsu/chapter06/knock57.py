from knock52 import load_features, load_labels, train_model
import numpy as np

def main():
    # 特徴量の読み込み
    train_features = load_features("train.feature.txt")

    # ラベルの読み込み
    train_labels = load_labels("train.txt")

    # モデルの学習
    model = train_model(train_features, train_labels)

    # 特徴量の重みを取得
    weights = model.coef_

    # 特徴量の名前を取得
    feature_names = np.array(range(weights.shape[1]))

    # カテゴリごとに重みの高い特徴量トップ10と重みの低い特徴量トップ10を表示
    for i, category in enumerate(model.classes_):
        print(f"カテゴリ: {category}")

        # 重みの高い特徴量トップ10
        top_indices = np.argsort(weights[i])[::-1][:10]
        top_features = feature_names[top_indices]
        top_weights = weights[i][top_indices]
        print("重みの高い特徴量トップ10:")
        for feature, weight in zip(top_features, top_weights):
            print(f"特徴量 {feature}: {weight:.4f}")

        # 重みの低い特徴量トップ10
        bottom_indices = np.argsort(weights[i])[:10]
        bottom_features = feature_names[bottom_indices]
        bottom_weights = weights[i][bottom_indices]
        print("重みの低い特徴量トップ10:")
        for feature, weight in zip(bottom_features, bottom_weights):
            print(f"特徴量 {feature}: {weight:.4f}")

        print()

if __name__ == "__main__":
    main()