"""
1. カテゴリ名に「gram」が含まれているかどうかで、意味的アナロジーと文法的アナロジーのデータフレームを分ける
2. 各データフレームについて、正解（True_wordとPred_wordが一致）をカウント
3. 各アナロジータスクの正解率を計算し、結果を出力

"""

import pandas as pd

# アナロジー結果の読み込み
df = pd.read_csv("Anology_example.csv")

# 意味的アナロジーと文法的アナロジーのデータフレームに分割
df_se = df[df["Category"].str.contains("gram")]
df_sy = df[~df["Category"].str.contains("gram")]

# 正解数をカウント
df_se_true = df_se[df_se["True_word"] == df_se["Pred_word"]]
df_sy_true = df_sy[df_sy["True_word"] == df_sy["Pred_word"]]

# 正解率の計算
semantic_accuracy = len(df_se_true) / len(df_se)
syntactic_accuracy = len(df_sy_true) / len(df_sy)

print(f"semantic analogy: {semantic_accuracy:.2%}")
print(f"syntactic analogy: {syntactic_accuracy:.2%}")

# semantic analogy: 0.7400468384074942
# syntactic analogy: 0.7308602999210734
