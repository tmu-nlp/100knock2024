"""
t-SNEは機械学習の教師なし学習の中のひとつで、次元削減を行うアルゴリズム
t-SNEはPCAなどの可視化手法とは異なり、線形では表現できない関係も学習して次元削減を行える利点がある

t-SNEではあるデータ点とあるデータ点の近さを同時確立として表現
元データと次元削減後のデータの近さをKLダイバージェンスを最小化することで次元削減の学習を行います。
KLダイバージェンスとは2つの確率分布の間の異なり具合を測るものになっている

実際には削減後のデータを乱数で初期化し、KL divergenceを勾配法で最小化
"""
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
X = np.array(countryVec)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
pred = KMeans(n_clusters = 5, random_state=777).fit_predict(X)

plt.figure(figsize=(15, 13))
col_list = ["Blue", "Red", "Green", "Black"]
for x, name, km in zip(X_tsne, countryName, pred):
    plt.plot(x[0], x[1], color = col_list[km-1], marker="o")
    plt.annotate(name, xy=(x[0], x[1]))#plt.annotate関数を使用して、データ点の位置に国名の注釈を追加 xy=(x[0], x[1])で注釈の位置を指定
plt.title("T-SNE")
plt.savefig("TSNE.png")
plt.show()