#問22
#記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．


#replace()でカテゴリ名以外の余計な文字列を削除し出力
#str.replace()メソッド：指定した文字列を別の文字列に置き換える。
import pandas as pd
import re 

filename = "jawiki-country.json"

j_data = pd.read_json(filename, lines =True)
df = j_data
uk_df = df[df["title"]=="イギリス"]

for text in uk_df[0].split("\n"):
    if re.search("Category", text):
        text = text.replace("[[Category:", "").replace("|*]]", "").replace("]]", "")
        print(text)