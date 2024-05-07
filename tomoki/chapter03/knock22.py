#No22(カテゴリ名の抽出)
import pandas as pd
import re
df=pd.read_json("jawiki-country.json.gz",lines=True)
D_8=df[df["title"]=="イギリス"]
D_8=D_8["text"].values
#No21の問題で作成したものから、不必要なものを取り除く
#[[Category:イギリス|*]]
for text in D_8[0].split("\n"):
    if re.search("Category", text):
        text2=re.sub("Category:|[[]|]","",text)
        text2=text2.replace("|","").replace("*","")
        print(text2)
        

        