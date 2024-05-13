#問23
#記事中に含まれるセクション名とそのレベル（例えば”== セクション名 ==”なら1）を表示せよ．


#セクション構造

#イギリスデータの要素を空白文字で区切り、反復させる 
#正規表現のメタ文字を使用して、セクション名をサーチ
#“=”の数ごとのレベルを取得し、セクション名とレベルを出力

#シーケンス型共通のcount()メソッド：引数に指定した要素をカウントする
import pandas as pd
import re 

filename = "jawiki-country.json"

j_data = pd.read_json(filename, lines =True)
df = j_data
uk_df = df[df["title"]=="イギリス"]

for text in uk_df[0].split("\n"):
    if re.search("^=+.*=+$", text):
        num = text.count("=") / 2 - 1
        print(text.replace("=", ""), int(num))