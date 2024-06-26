import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk import stem

def load_data(filepath):
    return pd.read_csv(filepath, sep="\t", header=None, names=["CATEGORY", "TITLE"])

def preprocess_text(text):
    # 記号の削除
    text = re.sub(r'[\"\'.,:;\(\)#\|\*\+\!\?#$%&/\]\[\{\}]', '', text)
    # ' - 'みたいなつなぎ文字を削除
    text = re.sub('\\s-\\s', ' ', text)
    # 数字の正規化(全部0にする)
    text = re.sub('[0-9]+', '0', text)
    # 小文字化
    text = text.lower()
    # ステミングで語幹だけ取り出す
    stemmer = stem.PorterStemmer()
    words = [stemmer.stem(word) for word in text.split()]
    return ' '.join(words)

def extract_features(df):
    # 前処理の適用
    df["TITLE"] = df["TITLE"].apply(preprocess_text)

    # TF-IDFを計算
    vectorizer = TfidfVectorizer(min_df=10, ngram_range=(1, 2))
    features = vectorizer.fit_transform(df["TITLE"])
    return features, vectorizer

def save_features(features, filepath):
    # 特徴量を保存
    features.to_csv(filepath, sep="\t", index=False, header=False)

def main():
    # データの読み込み
    train_df = load_data("train.txt")
    valid_df = load_data("valid.txt")
    test_df = load_data("test.txt")

    # データの結合
    df = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)

    # 特徴量の抽出
    features, vectorizer = extract_features(df)
    feature_df = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())

    # 特徴量の分割
    train_features = feature_df.iloc[:len(train_df), :]
    valid_features = feature_df.iloc[len(train_df):len(train_df) + len(valid_df), :]
    test_features = feature_df.iloc[len(train_df) + len(valid_df):, :]

    # 特徴量の保存
    save_features(train_features, "train.feature.txt")
    save_features(valid_features, "valid.feature.txt")
    save_features(test_features, "test.feature.txt")

    print(train_features.shape)

if __name__ == "__main__":
    main()