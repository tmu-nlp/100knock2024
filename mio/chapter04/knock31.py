#31. 動詞の表層形
#動詞の表層形をすべて抽出せよ．

#方針１：空の集合surface_verbをつくる
#　　　　（なぜsetなのか？→重複を避けるため）
#方針２：問30で作成したresultの中身について、品詞が動詞なら表層形をsetに格納

import knock30


#作業１
 #set()：集合（重複しない要素のあつまり）を表す/組み込みのデータ型
verb_surface = set()

#作業２
for sentence in knock30.morph_result:
  for word in sentence:
    if word["pos"] == "動詞":
      verb_surface.add(word["surface"])

print(verb_surface)

