#27.内部リンクの除去
#26の処理に加えて，テンプレートの値からMediaWikiの内部リンクマークアップを除去し，テキストに変換せよ

for text in uk_df[0].split("\n"):
    if re.search("\|(.+?)\s=\s*(.+)", text):
        match_txt = re.search("\|(.+?)\s=\s*(.+)", text)
        dic[match_txt[1]] = match_txt[2]
    match_sub = re.sub("\'{2,}(.+?)\'{2,}", "\\1", text)
    match_sub = re.sub("\[\[(.+?)\]\]", "\\1", match_sub)
    print(match_sub)

"""
[ [記事名] ]->記事名
[ [記事名|表示文字 ] ]->表示文字

dfの中身を一行ずつ取り出し、text変数に代入

｜key = valueのパターンの中身があれば、text変数に保存
 match_txt[1]` にはkeyの部分、`match_txt[2]` にはvalueの部分が入る、
  このkey-valueペアを辞書に蓄積

re.sub()で正規表現を使って文字列の置換を行う。

textの中から、2個以上の単一引用符 ' で囲まれた部分を探し出す.
  その部分を、括弧の内部の文字列だけに置換する
二重括弧 [[ で囲まれた部分を検索し、括弧部分を除去してページ名だけを抽出

"""
