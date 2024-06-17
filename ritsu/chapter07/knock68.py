import numpy as np
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

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

    # Ward法による階層型クラスタリング
    Z = linkage(countries_vec, method='ward')

    # デンドログラムの可視化
    plt.figure(figsize=(20, 10))
    dendrogram(Z, labels=countries, leaf_rotation=90, leaf_font_size=8)
    plt.tight_layout()
    plt.savefig('ward_dendrogram.png')
    plt.show()

if __name__ == '__main__':
    main()