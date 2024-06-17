"""
1. Google Newsデータセットの学習済みモデルをロード
2. questions-words.txtファイルを読み込み、カテゴリごとに単語アナロジーの評価データをDataFrameに整理
3. 各アナロジー問題について、ベクトル計算を行い、最も類似する単語とその類似度を取得
4. 結果をAnology_example.csvファイルに保存

"""

import pandas as pd
from gensim.models import KeyedVectors

# モデルのロード
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

# 単語アナロジーの評価データの読み込み
with open("questions-words.txt", "r") as f:
    lines = f.readlines()
    word1 = []
    word2 = []
    word3 = []
    word4 = []
    df = pd.DataFrame()
    for i, line in enumerate(lines, 1):
        if line[0] == ":":
            if len(word1) != 0:
                df_temp = pd.DataFrame({"1st_word": word1, "2nd_word": word2,
                                        "3rd_word": word3, "True_word": word4})
                df_temp["Category"] = category.replace("\n", "").replace(":", "")
                df = pd.concat([df, df_temp])
                word1 = []
                word2 = []
                word3 = []
                word4 = []
            category = line
        else:
            word = line.split(" ")
            word1.append(word[0])
            word2.append(word[1])
            word3.append(word[2])
            word4.append(word[3].replace("\n", ""))
            if len(lines) == i:
                df_temp = pd.DataFrame({"1st_word": word1, "2nd_word": word2,
                                        "3rd_word": word3, "True_word": word4})
                df_temp["Category"] = category.replace("\n", "").replace(":", "")
                df = pd.concat([df, df_temp])

df = df.reset_index(drop=True)

# アナロジー計算関数の定義
def calculate_word_vec(row):
    result = model.most_similar(positive=[row["2nd_word"], row["3rd_word"]],
                                negative=[row["1st_word"]], topn=1)[0]
    return result[0], result[1]

# アナロジー計算の実行
df[["Pred_word", "similarity"]] = df.apply(calculate_word_vec, axis=1, result_type="expand")

# 結果の保存
output_path = "[PATH]/Anology_example.csv"
df.to_csv(output_path, index=False)

print(f"結果が{output_path}に保存された")
df
