import numpy as np
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    # GoogleNews-vectors-negative300.bin.gzの読み込み
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

    # 国名の取得
    countries = set()
    with open('questions-words-add.txt', 'r') as f:
        for line in f:
            line = line.split()
            if line[0] in ['capital-common-countries', 'capital-world']:
                countries.add(line[2])
            elif line[0] in ['currency', 'gram6-nationality-adjective']:
                countries.add(line[1])
    countries = list(countries)

    # 単語ベクトルの取得
    countries_vec = [model[country] for country in countries]

    # k-meansクラスタリング
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(countries_vec)

    # t-SNEによる次元削減
    tsne = TSNE(n_components=2, random_state=64)
    X_reduced = tsne.fit_transform(np.array(countries_vec))

    # 可視化
    plt.figure(figsize=(10, 10))
    for x, country, color in zip(X_reduced, countries, kmeans.labels_):
        plt.text(x[0], x[1], country, color=f'C{color}')
    plt.xlim([-12, 15])
    plt.ylim([-15, 15])
    plt.savefig('t-SNE_countries_with_kmeans.png')
    plt.show()

if __name__ == '__main__':
    main()