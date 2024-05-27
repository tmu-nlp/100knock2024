#26.強調マークアップの除去
#25の処理時に，テンプレートの値からMediaWikiの強調マークアップ（弱い強調，強調，強い強調のすべて）を除去してテキストに変換せよ

for text in uk_df[0].split("\n"):
    if re.search("\|(.+?)\s=\s*(.+)", text):
        match_txt = re.search("\|(.+?)\s=\s*(.+)", text)
        dic[match_txt[1]] = match_txt[2]
    match_sub = re.sub("\'{2,}(.+?)\'{2,}", "\\1", text)
    print(match_sub)