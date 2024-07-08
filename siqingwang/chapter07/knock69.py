# knock69

from sklearn.manifold import TSNE

tsne = TSNE()
tsne.fit(countries_vec)

plt.figure(figsize=(15, 15), dpi=300)
plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1])
for (x, y), name in zip(tsne.embedding_, countries):
    plt.annotate(name, (x, y))
plt.show()