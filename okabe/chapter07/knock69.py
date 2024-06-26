# knock69
# ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ

import bhtsne
import numpy as np
import matplotlib.pyplot as plt
from knock67 import countries, countries_vec

embedded = bhtsne.tsne(np.array(countries_vec).astype(np.float64), dimensions=2, rand_seed=123)
plt.figure(figsize=(10, 10))
plt.scatter(np.array(embedded).T[0], np.array(embedded).T[1])
for (x, y), name in zip(embedded, countries):
    plt.annotate(name, (x, y))
#plt.savefig("output69.png")
plt.show()