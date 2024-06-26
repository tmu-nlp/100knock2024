import knock30
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import defaultdict #存在チェックが不要
from collections import OrderedDict #順番ありの辞書にする

cat_word_dict = defaultdict(lambda: 0) #lambdaは初期化の関数

#各文に”猫”が含まれているかの判別

#文の数だけ要素があるリスト作成
cat_bool = [0] * len(knock30.sentence_list) #knock30のsentence_listを利用

#cat_boolについて、”猫”が含まれている文のインデックスの要素を1にする
for i, line in enumerate(knock30.sentence_list):
    for morph in line:
        if morph["surface"] == "猫":
            cat_bool[i] = 1

#猫」とよく共起する（共起頻度が高い）10語とその出現頻度を辞書にする
for i, line in enumerate(knock30.sentence_list):
    for morph in line:
        if cat_bool[i] == 1 and (morph["surface"] != "猫" and morph["pos"] != "記号"): #猫がある文章の猫と記号以外の単語を抜き出す。
            cat_word_dict[morph["surface"]] += 1 #出た回数をカウント

sort_word_dict = OrderedDict(sorted(cat_word_dict.items(), key = lambda x:x[1], reverse=True)[:10]) #cat_word_dictのキーは単語、値は出現回数、ソートの方法は出現回数で降順

print(sort_word_dict)

#プロット

label = []
data = []

for k,v in sort_word_dict.items():
    label.append(k)
    data.append(v)
x = [1,2,3,4,5,6,7,8,9,10]
# print(label)
plt.bar(x,data)
plt.xticks(x,label)
plt.xlabel("語")
plt.ylabel("出現頻度")
plt.title("猫と共起頻度の高い上位10語")
#plt.savefig("knock37.png")