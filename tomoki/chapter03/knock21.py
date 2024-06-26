#No21(カテゴリ名を含む行を抽出)
import pandas as pd
#re python正規表現モジュール(標準モジュール)
import re
df=pd.read_json("jawiki-country.json.gz",lines=True)
D_8=df[df["title"]=="イギリス"]
D_8=D_8["text"].values
#記事中でカテゴリ名を宣言している行は「Category:」を含む
#splitはstr型に使用できる。
for text in D_8[0].split("\n"):
    if re.search("Category", text):
        print(text)


#re.search・・・マッチする最初の場所を文字列全体から探す
#re.match・・・　文字列先頭から探す（文字列先頭とパターンがマッチするかどうかを調べる）