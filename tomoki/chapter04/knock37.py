#NO37(「猫」と共起頻度の高い上位10語)
#共起・・・自然言語処理の分野において、任意の文書や文において、ある文字列とある文字列が同時に出現することである。
from collections import Counter
import matplotlib.pyplot as plt
import japanize_matplotlib
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

text = []
for lis in result:
  #any関数　要素の中に一つでもTrueがあればTrueを返す
  #iterableの形にすること
  if any(d["base"] == "猫" for d in lis):
    for dic in lis:
      #猫以外の時、リストに追加する
      if dic["surface"] != "猫":
        text.append(dic["surface"])
      else:
        pass
count = Counter(text)
target = list(zip(*count.most_common( 10 ))) 
plt.bar(*target) 
#tkinterのinstallをするとplt.show()がうまく利用できる
plt.show()