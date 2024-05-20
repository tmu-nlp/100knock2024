#問28
#27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，国の基本情報を整形せよ

#該当するマークアップをre.sub()で削除

import pandas as pd
import re 

filename = "jawiki-country.json"

j_data = pd.read_json(filename, lines =True)
df = j_data
uk_df = df[df["title"]=="イギリス"]
dic = {}
for text in uk_df[0].split("\n"):
    if re.search("\|(.+?)\s=\s*(.+)", text):
        match_sub = re.search("\|(.+?)\s=\s*(.+)", text)
        dic[match_sub[1]] = match_sub[2]
    match_sub = re.sub("\'{2,}(.+?)\'{2,}", "\\1", text) #強調マークアップの削除
    match_sub = re.sub("\[\[(.+?)\]\]", "\\1", match_sub) #内部リンクマークアップの削除
    match_sub = re.sub("\[(.+?)\]", "\\1", match_sub) #外部リンクマークアップの削除
    match_sub = re.sub("\*+(.+?)", "\\1", match_sub) #*箇条書きの削除
    match_sub = re.sub("\:+(.+?)", "\\1", match_sub)#定義の箇条書きの削除
    match_sub = re.sub("\{\{(.+?)\}\}", "\\1", match_sub) #スタブなどを削除
    print(match_sub)