#67. k-meansクラスタリングPermalink
#国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．

import gensim.downloader as api
from sklearn.cluster import KMeans
import numpy as np

# Word2Vecモデルをダウンロード
model = api.load("word2vec-google-news-300")

# 国名のリスト（例として一部の国を使用）
countries = ["usa", "canada", "mexico", "brazil", "argentina", "uk", "france", "germany", "italy", "spain", 
             "russia", "china", "japan", "india", "australia", "egypt", "nigeria", "southafrica", "kenya", "morocco"]

# 国名の単語ベクトルを抽出
country_vectors = []
valid_countries = []

for country in countries:
    try:
        vector = model[country]
        country_vectors.append(vector)
        valid_countries.append(country)
    except KeyError:
        print(f"'{country}' not found in the model vocabulary.")

# NumPy配列に変換
country_vectors = np.array(country_vectors)

# K-meansクラスタリングを実行（k=5）
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(country_vectors)

# 結果を表示
for country, cluster in zip(valid_countries, cluster_labels):
    print(f"{country}: Cluster {cluster}")

# クラスタごとの国名をグループ化
clusters = {}
for country, cluster in zip(valid_countries, cluster_labels):
    if cluster not in clusters:
        clusters[cluster] = []
    clusters[cluster].append(country)

# クラスタごとの結果を表示
for cluster, countries in clusters.items():
    print(f"\nCluster {cluster}:")
    print(", ".join(countries))

"""
'southafrica' not found in the model vocabulary.
usa: Cluster 4
canada: Cluster 2
mexico: Cluster 4
brazil: Cluster 4
argentina: Cluster 4
uk: Cluster 4
france: Cluster 0
germany: Cluster 0
italy: Cluster 0
spain: Cluster 4
russia: Cluster 0
china: Cluster 1
japan: Cluster 0
india: Cluster 4
australia: Cluster 2
egypt: Cluster 1
nigeria: Cluster 3
kenya: Cluster 1
morocco: Cluster 1

Cluster 4:
usa, mexico, brazil, argentina, uk, spain, india

Cluster 2:
canada, australia

Cluster 0:
france, germany, italy, russia, japan

Cluster 1:
china, egypt, kenya, morocco

Cluster 3:
nigeria
"""
