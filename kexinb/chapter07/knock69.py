# task69. t-SNEによる可視化
# ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ．

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from knock67 import country_vectors, valid_country_names

# Apply t-SNE to reduce dimensions to 2D for visualization
tsne = TSNE(n_components=2, random_state=0)
country_vectors_2d = tsne.fit_transform(country_vectors)

# Create a scatter plot
plt.figure(figsize=(15, 10))
plt.scatter(country_vectors_2d[:, 0], country_vectors_2d[:, 1])

# Annotate the points with country names
for i, country in enumerate(valid_country_names):
    plt.annotate(country, (country_vectors_2d[i, 0], country_vectors_2d[i, 1]), fontsize=8)

plt.title('t-SNE Visualization of Country Word Vectors')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.savefig('output/ch7/knock69_out.png')

plt.show()
