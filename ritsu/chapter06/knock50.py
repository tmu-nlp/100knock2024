import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath, sep="\t", header=None, names=[
                     "ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
    return df

def filter_publishers(df, publishers):
    df = df[df["PUBLISHER"].isin(publishers)]
    return df

def shuffle_data(df):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def split_data(df):
    train_df, test_valid_df = train_test_split(
        df, test_size=0.2, stratify=df["CATEGORY"], random_state=42)
    valid_df, test_df = train_test_split(
        test_valid_df, test_size=0.5, stratify=test_valid_df["CATEGORY"], random_state=42)
    return train_df, valid_df, test_df

def save_data(train_df, valid_df, test_df):
    train_df[["CATEGORY", "TITLE"]].to_csv(
        "train.txt", sep="\t", index=False, header=False)
    valid_df[["CATEGORY", "TITLE"]].to_csv(
        "valid.txt", sep="\t", index=False, header=False)
    test_df[["CATEGORY", "TITLE"]].to_csv(
        "test.txt", sep="\t", index=False, header=False)

def print_category_counts(train_df, valid_df, test_df):
    print("学習データ")
    print(train_df["CATEGORY"].value_counts())
    print("検証データ")
    print(valid_df["CATEGORY"].value_counts())
    print("評価データ")
    print(test_df["CATEGORY"].value_counts())

def main():
    base_path = "./news+aggregator/"
    news_corpora_file = os.path.join(base_path, "newsCorpora.csv")

    df = load_data(news_corpora_file)

    publishers = ["Reuters", "Huffington Post",
                  "Businessweek", "Contactmusic.com", "Daily Mail"]
    df = filter_publishers(df, publishers)

    df = shuffle_data(df)

    train_df, valid_df, test_df = split_data(df)

    save_data(train_df, valid_df, test_df)

    print_category_counts(train_df, valid_df, test_df)

if __name__ == "__main__":
    main()