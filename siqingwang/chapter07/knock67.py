# knock67

import numpy as np
from sklearn.cluster import KMeans

countries = set()
with open('questions-words-add.txt', 'r') as f3:
  for line in f3:
    line = line.split()
    if line[0] in ['capital-common-countries', 'capital-world']:
      countries.add(line[2])
    elif line[0] in ['currency', 'gram6-nationality-adjective']:
      countries.add(line[1])
countries = list(countries)

countries_vec = [model[country] for country in countries]


# k-means
kmeans = KMeans(n_clusters=5)
kmeans.fit(countries_vec)
for i in range(5):
    cluster = np.where(kmeans.labels_ == i)[0]
    print('cluster', i)
    print(', '.join([countries[k] for k in cluster]))