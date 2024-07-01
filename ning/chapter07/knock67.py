from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# モデルのロード
model = KeyedVectors.load_word2vec_format("[PATH]/GoogleNews-vectors-negative300.bin", binary=True)

# 国名リストの作成
countries = ["Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina", "Armenia", "Australia",
             "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", 
             "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia", "Botswana", "Brazil", "Brunei", "Bulgaria", 
             "Burkina_Faso", "Burundi", "Cabo_Verde", "Cambodia", "Cameroon", "Canada", "Central_African_Republic",
             "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa_Rica", "Croatia", "Cuba", "Cyprus", 
             "Czech_Republic", "Denmark", "Djibouti", "Dominica", "Dominican_Republic", "Ecuador", "Egypt", 
             "El_Salvador", "Equatorial_Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", 
             "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", 
             "Guinea_Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", 
             "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", 
             "Korea", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", 
             "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", 
             "Malta", "Marshall_Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", 
             "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands",
             "New_Zealand", "Nicaragua", "Niger", "Nigeria", "North_Macedonia", "Norway", "Oman", "Pakistan", "Palau",
             "Panama", "Papua_New_Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", 
             "Romania", "Russia", "Rwanda", "Saint_Kitts_and_Nevis", "Saint_Lucia", "Saint_Vincent_and_the_Grenadines",
             "Samoa", "San_Marino", "Sao_Tome_and_Principe", "Saudi_Arabia", "Senegal", "Serbia", "Seychelles", 
             "Sierra_Leone", "Singapore", "Slovakia", "Slovenia", "Solomon_Islands", "Somalia", "South_Africa", 
             "South_Sudan", "Spain", "Sri_Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Taiwan", 
             "Tajikistan", "Tanzania", "Thailand", "Timor_Leste", "Togo", "Tonga", "Trinidad_and_Tobago", "Tunisia", 
             "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United_Arab_Emirates", "United_Kingdom", 
             "United_States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican_City", "Venezuela", "Vietnam", "Yemen", 
             "Zambia", "Zimbabwe"]

# 国名の単語ベクトルを抽出
country_vectors = []
valid_countries = []
for country in countries:
    if country in model:
        country_vectors.append(model[country])
        valid_countries.append(country)

country_vectors = np.array(country_vectors)

# k-meansクラスタリングの実行
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(country_vectors)
labels = kmeans.labels_

# 結果をデータフレームにまとめる
df = pd.DataFrame({'Country': valid_countries, 'Cluster': labels})

print(df)

# 反省：国名リストをファイルとしてimportするべき