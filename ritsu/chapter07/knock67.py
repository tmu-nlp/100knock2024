import numpy as np
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

def main():
    # GoogleNews-vectors-negative300.bin.gzの読み込み
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

    # 国名の取得
    countries = set()
    with open('questions-words-add.txt', 'r') as f:
        for line in f:
            line = line.split()
            # capital-common-countries, capital-world, currency, gram6-nationalを用いて国名を取得
            if line[0] in ['capital-common-countries', 'capital-world']:
                countries.add(line[2])
            elif line[0] in ['currency', 'gram6-nationality-adjective']:
                countries.add(line[1])
    countries = list(countries)

    # 単語ベクトルの取得
    countries_vec = [model[country] for country in countries]

    # k-meansクラスタリング
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(countries_vec)

    # クラスタリング結果の出力
    for i in range(n_clusters):
        cluster = np.where(kmeans.labels_ == i)[0]
        print(f'Cluster {i}:')
        print(', '.join([countries[k] for k in cluster]))
        print()

if __name__ == '__main__':
    main()

"""
Cluster 0:
Denmark, Japan, Austria, Spain, Tuvalu, Malaysia, Belgium, Portugal, Morocco, England, Oman, Malta, Netherlands, Samoa, Sweden, Vietnam, Europe, Nepal, Iceland, Thailand, Italy, Ireland, Greenland, Indonesia, Australia, China, India, Philippines, USA, Germany, Fiji, Laos, Switzerland, Canada, Korea, Taiwan, Finland, Qatar, Bahrain, Liechtenstein, Cambodia, Bhutan, France, Norway, Bangladesh

Cluster 1:
Uganda, Namibia, Nigeria, Rwanda, Niger, Zimbabwe, Mozambique, Mali, Burundi, Algeria, Mauritania, Madagascar, Gabon, Senegal, Tunisia, Kenya, Gambia, Malawi, Liberia, Zambia, Angola, Guinea, Botswana, Ghana

Cluster 2:
Dominica, Colombia, Brazil, Cuba, Peru, Jamaica, Uruguay, Nicaragua, Suriname, Belize, Chile, Ecuador, Honduras, Argentina, Guyana, Venezuela, Mexico, Bahamas

Cluster 3:
Lebanon, Libya, Sudan, Syria, Israel, Pakistan, Iran, Egypt, Afghanistan, Jordan, Iraq, Somalia, Eritrea

Cluster 4:
Croatia, Albania, Latvia, Ukraine, Cyprus, Azerbaijan, Poland, Montenegro, Greece, Russia, Macedonia, Georgia, Kyrgyzstan, Estonia, Uzbekistan, Hungary, Turkmenistan, Turkey, Armenia, Belarus, Lithuania, Moldova, Bulgaria, Tajikistan, Slovenia, Serbia, Kazakhstan, Romania, Slovakia
"""