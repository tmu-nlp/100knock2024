from gensim.models import KeyedVectors
from scipy.stats import spearmanr
import pandas as pd

# モデルのロード
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# WordSimilarity-353の評価データを読み込み 
df = pd.read_csv("combined.csv")

# 単語ベクトルにより計算される類似度を計算
Cossim = lambda X: model.similarity(X["Word 1"], X["Word 2"])
df["WordVec_Similarity"] = df.apply(Cossim, axis=1)

# スピアマン相関係数の計算
correlation, pvalue = spearmanr(df["WordVec_Similarity"], df["Human (mean)"])

print("スピアマン相関係数:", correlation)
print("p値:", pvalue)

# スピアマン相関係数: 0.7000166486272194
# p値: 2.86866666051422e-53

