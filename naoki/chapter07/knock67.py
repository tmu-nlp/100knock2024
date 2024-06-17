import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import pickle


with open("drive/MyDrive/word2vec.pkl", "rb") as f:
    model = pickle.load(f)
df = pd.read_csv('drive/MyDrive/questions-words.txt',sep=' ',skiprows=1)
df = df.dropna()
df = df.reset_index()
df_country = df.iloc[:5030]
df_country.columns = ['index','word1','word2','word3','word4']
country = list(set(df_country.loc[:,'word4'].values))
countryVec = []
countryName = []
for c in country:
    countryVec.append(model[c])
    countryName.append(c)
#ndarray型に変換することで計算ができるようになる
X = np.array(countryVec)
km = KMeans(n_clusters=5, random_state=777)
y_km = km.fit_predict(X)
dic = {}
for num, name in zip(y_km, countryName):
    dic.setdefault(num, []).append(name)
dic