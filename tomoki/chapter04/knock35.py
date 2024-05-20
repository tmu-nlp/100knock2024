#NO35(単語の出現頻度)
#pythonの標準ライブラリ
#各要素の出現回数を一括で取得したい場合に利用するとよい
#(要素, 出現回数)というように値を返す
#collections.Counterクラスは辞書型のサブクラス。よって辞書型と同じように扱える。
from collections import Counter
 
result = []
sentence = []
with open("neko.txt.mecab") as f:
  for line in f:
    l1 = line.split("\t")
    if len(l1) == 2:
      l2 = l1[1].split(",")
    #list out of rangeを防ぐ(エラーが起きたときは、基本形に表層形を利用する)
      try:
        #l1は表層形のみを含む　l2は[0:品詞 1:品詞細分類1・・・のようになっている]
        sentence.append({"surface": l1[0], "base": l2[7], "pos": l2[0], "pos1": l2[1]})
      except IndexError:
        sentence.append({"surface": l1[0], "base": l1[0], "pos": l2[0], "pos1": l2[1]})

      if l2[1] == "句点":
        result.append(sentence)
        sentence = []
result

text2 = []
#resultはネスト(listのなかにdictという多重構造)になっているため、2回for文を回す必用がある。
for lis in result:
  for dic in lis:
    #""以外の時、リストに追加する
    if dic["surface"] != "":
      text2.append(dic["surface"])
 
count = Counter(text2)
#most_common()メゾット　(要素, 出現回数)という形のタプルを出現回数順に並べたリストを返す。
print(count.most_common())