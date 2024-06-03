#３５．単語の出現頻度
#文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．

#方針１：記号以外の単語の基本形が現れたらその単語の頻度を1増やす
#　　　　　　　　　　　なければの単語の頻度を0にする
#方針２：出現頻度の高い順にソート

import knock30


frequency = {}

 #作業１
for sentence in knock30.morph_results:
  for word in sentence:
    #記号は単語には含めない
    if word["pos"] == "記号":
      pass
    elif word["base"] in frequency:
      frequency[word["base"]]+= 1
    else:
      frequency[word["base"]] = 0

#組み込み関数sorted(リスト/辞書名, key=lambda x: keyにしたい要素)
# ※lambda：無名関数/ここでは「要素xを受け取り、x[1]を返す」式

#sort()とsorted()の違い：元の辞書/リストが変更されるか？
#sort()：リスト型のメソッド/既存のリスト自体をソート（破壊的）
#sorted()：組み込み関数/既存のリストをソートしたものを新たに生成（非破壊的）

#items()：キーと値の組み合わせを取得

#作業２
frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

print(frequency)