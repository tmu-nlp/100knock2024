from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd

# モデルのロード
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# 国名の読み込みと変換
df = pd.read_csv("[PATH]/country_map.csv")
df["Country"] = df["Country"].replace(["United Arab Emirates", "United Kingdom", "United States of America", 
                                       "Saudi Arabi", "Syrian Arab Republic", "Czech Republic", 
                                       "Vatican City State", "Russian Federation"],
                                      ["UAE", "UK", "USA", "SAU", "SYR", "Czech", "Vatican", "Russia"])
country_name = df["Country"].tolist()
country_name.append("Japan")

# 国名の単語ベクトルを抽出
vec_list = []
country_name_fix = []
for name in country_name:
    try:
        vec = np.array(model.get_vector(name), "float64")
        vec_list.append(vec)
        country_name_fix.append(name)
    except:
        pass

# 階層型クラスタリングの実行
Z = linkage(vec_list, 'ward')

# デンドログラムのプロット
fig = plt.figure(figsize=(20, 10))
dn = dendrogram(Z, labels=country_name_fix)
plt.show()

# 結果をファイルに保存
fig.savefig("dendrogram.pdf")
