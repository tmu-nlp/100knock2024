#問27
#26の処理に加えて，テンプレートの値からMediaWikiの内部リンクマークアップを除去し，テキストに変換せよ

#26.で協調マークアップの除去を行ったように、内部リンクも除去して出力
#re.sub()を用いて、内部リンクマークを(.+?)に置換する。つまり、内部リンクマークを削除
import pandas as pd
import re 

filename = "jawiki-country.json"

j_data = pd.read_json(filename, lines =True)
df = j_data
uk_df = df[df["title"]=="イギリス"]

dic = {}
for text in uk_df[0].split("\n"):
    if re.search("\|(.+?)\s=\s*(.+)", text):
        match_txt = re.search("\|(.+?)\s=\s*(.+)", text)
        dic[match_txt[1]] = match_txt[2]
    match_sub = re.sub("\'{2,}(.+?)\'{2,}", "\\1", text)
    print(match_sub)