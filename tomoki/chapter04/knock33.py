#NO33(AのB)
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

#集合型を作る
se = set()
for line in result:
  for i in range(len(line)):
   #名詞＋"の"+名詞の形を見つけたら、seに加える
   if line[i]["pos"] == "名詞" and line[i + 1]["surface"] == "の" and line[i + 2]["pos"] == "名詞":
     se.add(line[i]["surface"] + line[i + 1]["surface"] + line[i + 2]["surface"])
 
print(se)