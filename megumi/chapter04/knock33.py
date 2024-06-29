#33.「AのB」
#2つの名詞が「の」で連結されている名詞句を抽出せよ．

import knock30
result=knock30.parse_neko()




#空のsetを作成する
se = set()

#問30で作成したresultを反復処理させる。
for line in result:
  for i in range(len(line)):
   #bool演算子(not,and,or)を使用して、条件分岐を行う。
   if line[i]["pos"] == "名詞" and line[i + 1]["surface"] == "の" and line[i + 2]["pos"] == "名詞":
    #重複を避けるため、条件にマッチした要素をsetに追加していく。
      se.add(line[i]["surface"] + line[i + 1]["surface"] + line[i + 2]["surface"])

print(se)

"""
出力結果
'初対面の人', '事蹟の三', '天地の間', 
'下女の顔', 'ここの細君', '得意のよう', 
'貧乏性の男', '君の悪口', '人の所有', 
'屋の大将', '窮措大の家', '鼻の在所', 
'馬鹿の相談', '吾輩のため'
"""