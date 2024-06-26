# task67. k-meansクラスタリング
# 国名に関する単語ベクトルを抽出し，k-meansクラスタリングをクラスタ数k=5として実行せよ

import pycountry
import pickle
import numpy as np
from sklearn.cluster import KMeans

# Get a list of country names
country_names = [country.name for country in pycountry.countries]

with open("output/ch7/word2vec.pkl", "rb") as f:
    model = pickle.load(f)
    # Extract word vectors for the country names
    country_vectors = []
    valid_country_names = []
    for country in country_names:
        if country in model:
            country_vectors.append(model[country])
            valid_country_names.append(country)

    # Ensure country_vectors is a numpy array for clustering
    country_vectors = np.array(country_vectors)

    # Perform k-means clustering with k=5
    kmeans = KMeans(n_clusters=5, random_state=42).fit(country_vectors)

    # Get the cluster labels
    labels = kmeans.labels_
    
    # Create a dictionary to hold the countries in each cluster
    clusters = {i: [] for i in range(5)}

    for country, label in zip(valid_country_names, labels):
        clusters[label].append(country)

if __name__ == "__main__":
    # Print the clusters
    for cluster_id, countries in clusters.items():
        print(f"Cluster {cluster_id}:")
        print(", ".join(countries))
        print()

'''
Cluster 0:
Botswana, Ghana, Kenya, Lesotho, Mozambique, Malawi, Namibia, 
Uganda, Zambia, Zimbabwe

Cluster 1:
Aruba, Andorra, Argentina, Austria, Belgium, Brazil, Canada, 
Switzerland, Curaçao, Germany, Denmark, Spain, Finland, France, Guernsey, 
Gibraltar, Guadeloupe, Greenland, Ireland, Iceland, Israel, Italy, Jersey, 
Jordan, Liechtenstein, Luxembourg, Morocco, Monaco, Malta, Martinique, Mayotte,
Netherlands, Norway, Portugal, Paraguay, Réunion, Sweden, Türkiye, Uruguay

Cluster 2:
Albania, Armenia, Azerbaijan, Bulgaria, Belarus, Cyprus, Czechia, Estonia, 
Georgia, Greece, Croatia, Hungary, Kazakhstan, Kyrgyzstan, Lithuania, Latvia, 
Montenegro, Poland, Romania, Serbia, Slovakia, Slovenia, Tajikistan, Turkmenistan, 
Ukraine, Uzbekistan

Cluster 3:
Afghanistan, Anguilla, Antarctica, Australia, Bangladesh, Bahrain, Bahamas, 
Belize, Bermuda, Barbados, Bhutan, Chile, China, Colombia, Cuba, Dominica, 
Ecuador, Fiji, Grenada, Guatemala, Guam, Guyana, Honduras, Indonesia, India, 
Iraq, Jamaica, Japan, Cambodia, Kiribati, Kuwait, Macao, Maldives, Mexico, 
Myanmar, Mongolia, Montserrat, Mauritius, Malaysia, Nicaragua, Niue, Nepal, Nauru, 
Oman, Pakistan, Panama, Pitcairn, Peru, Philippines, Palau, Qatar, Singapore, 
Suriname, Seychelles, Thailand, Tokelau, Tonga, Tuvalu, Vanuatu, Samoa

Cluster 4:
Angola, Burundi, Benin, Cameroon, Congo, Comoros, Djibouti, Algeria, Egypt, 
Eritrea, Ethiopia, Gabon, Guinea, Gambia, Haiti, Lebanon, Liberia, Libya, 
Madagascar, Mali, Mauritania, Niger, Nigeria, Rwanda, Sudan, Senegal, Somalia, 
Chad, Togo, Tunisia, Yemen
'''
