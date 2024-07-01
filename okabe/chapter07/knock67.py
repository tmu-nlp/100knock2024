'''
67. k-meansクラスタリング
国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ．
'''
from knock60 import model
from sklearn.cluster import KMeans
import numpy as np


# 国名の取得
countries = set()
with open('questions-words-add.txt') as f:
  for line in f:
    line = line.split()
    if line[0] in ['capital-common-countries', 'capital-world']:
      countries.add(line[2])
    elif line[0] in ['currency', 'gram6-nationality-adjective']:
      countries.add(line[1])
countries = list(countries)

# 単語ベクトルの取得
countries_vec = [model[country] for country in countries]

# k-meansクラスタリング
kmeans = KMeans(n_clusters=5)
kmeans.fit(countries_vec)
for i in range(5):
    cluster = np.where(kmeans.labels_ == i)[0]
    print('cluster', i)
    print(', '.join([countries[k] for k in cluster]))

"""
output:
cluster 0
Kenya, Uganda, Malawi, Zambia, Ghana, Angola, Niger, Botswana, Mauritania, Gambia, Madagascar, Senegal, Algeria, Gabon, Tunisia, Suriname, Burundi, Guinea, Namibia, Rwanda, Zimbabwe, Mali, Liberia, Guyana, Nigeria, Mozambique
cluster 1
Montenegro, Norway, Ireland, Iceland, Italy, Serbia, Liechtenstein, Germany, France, Lithuania, Latvia, Slovakia, Albania, Hungary, Netherlands, Greece, Turkey, Cyprus, Romania, Switzerland, Belgium, Denmark, Uruguay, Estonia, Malta, Bulgaria, Macedonia, Croatia, Portugal, Poland, Slovenia, Austria, Spain, Europe, Finland, Sweden
cluster 2
Iraq, Afghanistan, Libya, Israel, Iran, Egypt, Syria, Pakistan, Lebanon, Eritrea, Sudan, Somalia
cluster 3
Belarus, Russia, Armenia, Tajikistan, Kyrgyzstan, Kazakhstan, Turkmenistan, Azerbaijan, Moldova, Ukraine, Uzbekistan
cluster 4
Qatar, Nicaragua, Mexico, Honduras, Greenland, Bhutan, Nepal, Morocco, Colombia, Ecuador, Georgia, Bangladesh, Philippines, Taiwan, Samoa, Bahamas, India, Belize, China, Jamaica, Brazil, Cambodia, Peru, Bahrain, Malaysia, Tuvalu, Canada, Japan, Argentina, USA, Venezuela, Fiji, England, Oman, Vietnam, Australia, Laos, Korea, Indonesia, Chile, Cuba, Dominica, Thailand, Jordan
"""