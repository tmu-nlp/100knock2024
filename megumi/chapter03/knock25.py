#25.テンプレートの抽出
#記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，辞書オブジェクトとして格納せよ．

dic={}
for text in uk_df[0].split("\n"):
    if re.search("\|(.+?)\s=\s*(.+)", text):
        match_txt = re.search("\|(.+?)\s*=\s*(.+)", text)
        dic[match_txt[1]] = match_txt[2]
print(dic)