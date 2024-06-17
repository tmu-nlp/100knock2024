# task68. Ward法によるクラスタリング
# 国名に関する単語ベクトルに対し，Ward法による階層型クラスタリングを実行せよ．
# さらに，クラスタリング結果をデンドログラムとして可視化せよ．

import matplotlib.pyplot as plt
from knock67 import country_vectors, valid_country_names
from scipy.cluster.hierarchy import linkage, dendrogram

# Perform hierarchical clustering using Ward's method
Z = linkage(country_vectors, method='ward')

# Create a dendrogram to visualize clustering
plt.figure(figsize=(15, 10))
dendrogram(Z, labels=valid_country_names, leaf_rotation=90, leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram (Ward\'s method)')
plt.xlabel('Country')
plt.ylabel('Distance')
plt.savefig('output/ch7/knock68_out.png')

plt.show()