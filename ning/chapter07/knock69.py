from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd

# モデルのロード
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# 国名の読み込みと変換
df = pd.read_csv("country_map.csv")
df["Country"] = df["Country"].replace(["United Arab Emirates", "United Kingdom", "United States of America", 
                                       "Saudi Arabi", "Syrian Arab Republic", "Czech Republic", 
                                       "Vatican City State", "Russian Federation"],
                                      ["UAE", "UK", "USA", "SAU", "SYR", "Czech", "Vatican", "Russia"])
country_name = df["Country"].tolist()
country_name.append("Japan")

# 国名の単語ベクトルを抽出
vec_list = []
country_name_fix = []
for name in country_name:
    try:
        vec = np.array(model.get_vector(name), "float64")
        vec_list.append(vec)
        country_name_fix.append(name)
    except:
        pass

# t-SNEによる次元削減
vec_list = np.array(vec_list)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(vec_list)

# k-meansクラスタリングの実行
pred = KMeans(n_clusters=5, random_state=42).fit_predict(vec_list)

# t-SNEのプロット
plt.figure(figsize=(15, 13))
col_list = ["blue", "red", "green", "black", "purple"]
for X, name, km in zip(X_tsne, country_name_fix, pred):
    plt.scatter(X[0], X[1], color=col_list[km], marker="o")
    plt.annotate(name, xy=(X[0], X[1]))

plt.title("t-SNE Visualization of Country Vectors")
plt.savefig("TSNE.png")
plt.show()
