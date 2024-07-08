import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import dendrogram, linkage
countryVec = []
countryName = []
for c in country:
    countryVec.append(model[c])
    countryName.append(c)

X = np.array(countryVec)
linkage_result = linkage(X, method='ward', metric='euclidean')
plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor='w', edgecolor='k')
dendrogram(linkage_result, labels=countryName)
plt.show()