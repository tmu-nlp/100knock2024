#No28(MediaWikiマークアップの除去)
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
    match_sub = re.sub("\'{2,}(.+?)\'{2,}", "\\1", text)
    match_sub = re.sub("\[\[(.+?)\]\]", "\\1", match_sub)
    #{{}} を削除する
    match_sub = re.sub("\{{(.+?)}}","\\1",match_sub)
    #ex https://hdl.handle.net/11266/1231 大戦間期日・英造船業の企業金融
    #「:(何かの文字)(置換される内容)」という形を作る
    match_sub = re.sub("\:+(.+?)","\\1",match_sub)
    print(match_sub)