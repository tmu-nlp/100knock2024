#33. 「AのB」
#2つの名詞が「の」で連結されている名詞句を抽出せよ

#方針１：抽出した名詞句を格納するための空の集合を用意
#方針２：ある形態素の表層形が「の」、かつ、その一つ前後の形態素の品詞がともに名詞であるとき、
# 　　　 それらを連結して2つの名詞が「の」で連結されている名詞句の集合に追加

import knock30
#作業１
nounphrase_no = set()

#作業２
for sentence in knock30.morph_results:
  for i in range(1, len(sentence)):
   #
   if sentence[i-1]["pos"] == "名詞" and sentence[i]["surface"] == "の" and sentence[i + 1]["pos"] == "名詞":
     nounphrase_no.add(sentence[i-1]["surface"] + sentence[i]["surface"] + sentence[i + 1]["surface"])

print(nounphrase_no)
