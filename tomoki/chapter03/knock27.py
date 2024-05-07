#No27(内部リンクの除去)
#[[]]を取り除くことが今回の問題
import re
import pandas as pd
df=pd.read_json("jawiki-country.json.gz",lines=True)
D_8=df[df["title"]=="イギリス"]
D_8=D_8["text"].values
dict={}
for text in D_8[0].split("\n"):
    if re.search("\|(.+?)\s=\s*(.+)", text):
        match_txt = re.search("\|(.+?)\s=\s*(.+)", text)
        dict[match_txt[1]] = match_txt[2]
    match_sub = re.sub("\'{2,}(.+?)\'{2,}", "\\1", text)
    #26とやっていることはほぼ同じ。\\1をつかって(.+?)だけを表示するようにする。
    match_sub = re.sub("\[\[(.+?)\]\]", "\\1", match_sub)
    print(match_sub)