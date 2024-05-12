#No26(強調マークアップの除去)
#今回の問題では、シングルクォート２つ、３つもしくは５つで囲まれたものを除去すればよい
import re
import pandas as pd
df=pd.read_json("jawiki-country.json.gz",lines=True)
D_8=df[df["title"]=="イギリス"]
D_8=D_8["text"].values
dic={}
for text in D_8[0].split("\n"):
    if re.search("\|(.+?)\s=\s*(.+)", text):
        match_txt = re.search("\|(.+?)\s=\s*(.+)", text)
        dic[match_txt[1]] = match_txt[2]
    #「'{2,}」　'が２つ以上連続している時マッチする
    #「\\1」　（）で囲んだ部分を利用して文章を置換する
    match_sub = re.sub("\'{2,}(.+?)\'{2,}", "\\1", text)
    print(match_sub)