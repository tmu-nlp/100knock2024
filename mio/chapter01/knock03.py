#03. 円周率
#“Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.”という文を単語に分解し，
#各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ．
#方針：単語の文字数カウント作業を「,や.を文字数に数えないために除去した文字列」でループ→リストに格納
#replace("A", "B")AをBに置換
sentense ="Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
list= [len(word) for word in sentense.replace(",", "").replace(".", "").split()]
print(list)