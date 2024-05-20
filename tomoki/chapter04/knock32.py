import MeCab
result = []
sentence = []
with open("neko.txt.mecab") as f:
  for line in f:
    #(表層形)と（品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音）に分ける(_tでの分離)
    l1 = line.split("\t")
    if len(l1) == 2:
      #（品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音）を「,」で区切っていく
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
#集合型を作る
se = set()
for lis in result:
  for dic in lis:
    if dic["pos"] == "動詞":
      se.add(dic["base"])
 
print(se)