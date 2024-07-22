# knock68

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

plt.figure(figsize=(16, 9))
Z = linkage(countries_vec, method='ward')
dendrogram(Z, labels=countries)
plt.show()