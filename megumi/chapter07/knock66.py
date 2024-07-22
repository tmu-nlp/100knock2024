#66. WordSimilarity-353での評価
"""
The WordSimilarity-353 Test Collectionの評価データをダウンロードし，
単語ベクトルにより計算される類似度のランキングと，人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ．
"""
import gensim.downloader as api
import pandas as pd
from scipy.stats import spearmanr

# WordSimilarity-353 Test Collectionのデータを読み込む
df = pd.read_csv('./wordsim353/combined.csv')



# 単語ベクトルモデルをロード
model = api.load("word2vec-google-news-300")

# 単語ペアの類似度を計算
computed_similarities = []
human_similarities = df['Human (mean)'].tolist()

for i, row in df.iterrows():
    word1, word2 = row['Word 1'], row['Word 2']
    if word1 in model and word2 in model:
        similarity = model.similarity(word1, word2)
        computed_similarities.append(similarity)
    else:
        # モデルに単語がない場合は0を追加
        computed_similarities.append(0)

# スピアマン相関係数を計算
spearman_corr, _ = spearmanr(human_similarities, computed_similarities)
print(f"Spearman correlation coefficient: {spearman_corr}")

"""
Spearman correlation coefficient: 0.7000166486272194
"""
