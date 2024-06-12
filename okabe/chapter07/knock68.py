'''
68. Ward法によるクラスタリング
国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
さらに，クラスタリング結果をデンドログラムとして可視化せよ．
'''

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from knock67 import countries, countries_vec

plt.figure(figsize=(15, 5))
Z = linkage(countries_vec, method='ward')
dendrogram(Z, labels=countries)
#plt.savefig("output68.png")
plt.show()