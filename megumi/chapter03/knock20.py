#import requests
#url='https://nlp100.github.io/data/jawiki-country.json.gz'
#filename='jawiki-country.json.gz'
#urldata = requests.get(url).content
#with open(filename ,mode='wb') as f:
 #    .write(urldata)

#JSONデータの読み込み
#Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．
#問題21-29では，ここで抽出した記事本文に対して実行せよ．



import pandas as pd

filename = "jawiki-country.json"
j_data = pd.read_json(filename, lines =True)
df = j_data
uk_df = df[df["title"]=="イギリス"]
uk_df = uk_df["text"].values
print(uk_df)