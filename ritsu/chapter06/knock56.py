from knock52 import load_features, load_labels, train_model
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

def main():
    """
    適合率(precision): そのカテゴリに分類された中で、実際にそのカテゴリに属するサンプルの割合
    再現率(recall): 実際にそのカテゴリに属するサンプルの中で、正しく分類されたサンプルの割合
    F1スコア(f1-score): 適合率と再現率のバランスを表す指標で、適合率と再現率の調和平均として計算される
    マイクロ平均: 全てのサンプルに対して適合率，再現率，F1スコアを計算し，その平均を取る
    マクロ平均: カテゴリごとに適合率，再現率，F1スコアを計算し，その平均を取る

    """
    # 特徴量の読み込み
    train_features = load_features("train.feature.txt")
    test_features = load_features("test.feature.txt")

    # ラベルの読み込み
    train_labels = load_labels("train.txt")
    test_labels = load_labels("test.txt")

    # モデルの学習
    model = train_model(train_features, train_labels)

    # 評価データの予測
    test_predictions = model.predict(test_features)

    # 評価データ上で適合率，再現率，F1スコアを計測
    precision = precision_score(test_labels, test_predictions, average=None)
    recall = recall_score(test_labels, test_predictions, average=None)
    f1 = f1_score(test_labels, test_predictions, average=None)
    
    # データフレームの作成
    df = pd.DataFrame(columns=["適合率", "再現率", "F1値"], index=["b", "e", "t", "m", "macro-average", "micro-average"])
    df.loc["b":"m", "適合率"] = precision
    df.loc["b":"m", "再現率"] = recall
    df.loc["b":"m", "F1値"] = f1
    # averageを設定してマクロ平均とマイクロ平均を計算
    # マクロは各カテゴリの値を平均しクラスごとの性能を平等に評価
    # マイクロは全体の値を計算し全体的な性能を評価
    df.loc["macro-average", :] = [precision_score(test_labels, test_predictions, average="macro"), 
                                  recall_score(test_labels, test_predictions, average="macro"),
                                  f1_score(test_labels, test_predictions, average="macro")]
    df.loc["micro-average", :] = [precision_score(test_labels, test_predictions, average="micro"),
                                  recall_score(test_labels, test_predictions, average="micro"),
                                  f1_score(test_labels, test_predictions, average="micro")]
    
    # 結果の出力
    print(df)

if __name__ == "__main__":
    main()