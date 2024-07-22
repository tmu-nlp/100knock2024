# task66. WordSimilarity-353での評価
# The WordSimilarity-353 Test Collectionの評価データをダウンロードし，
# 単語ベクトルにより計算される類似度のランキングと，
# 人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ

import pickle
import csv
from scipy.stats import spearmanr

with open("output/ch7/word2vec.pkl", "rb") as f:
    model = pickle.load(f)

word1 = []
word2 = []
human_rank = []
model_rank = []

with open("data/combined.csv", "r") as txt:
    reader = csv.reader(txt)
    next(reader)  # skip header

    for line in reader:
        w1, w2, hr = line[0], line[1], line[2]
        word1.append(w1)
        word2.append(w2)
        human_rank.append(float(hr))

        sim = model.similarity(w1, w2)
        model_rank.append(sim)

# Calculate Spearman correlation
correlation, pvalue = spearmanr(human_rank, model_rank)
print(f"Spearman correlation: {correlation}, p-value: {pvalue}")

'''
Spearman correlation: 0.7000166486272194, p-value: 2.8686666605142608e-53
'''
