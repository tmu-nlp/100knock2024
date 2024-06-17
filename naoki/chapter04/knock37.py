"""
形態素解析
import MeCab
import unidic
mecab = MeCab.Tagger()
with open("[PATH]/neko.txt", "r") as f, open("[PATH]/neko.txt.mecab", "w") as f2:
    lines = f.readlines()
    for text in lines:
        result = mecab.parse(text)
        f2.write(result)
"""

with open("neko.txt.mecab", "r") as f:
    morphemes = []
    neko_list = []
    lines = f.readlines()
    for line in lines:
        neko_dic = {}
        suf = line.split("\t")
        if suf[0] == "EOS\n": 
            continue
        neko_dic["surface"] = suf[0]

        #suf[1]には名詞,普通名詞,副詞可能,,,,トキドキ,時々,時々,...
        try:
            temp = suf[1].split(',')
            neko_dic["base"] = temp[7]
            neko_dic["pos"] = temp[0]
            neko_dic["pos1"] = temp[1]
            neko_list.append(neko_dic)
        except:
            continue

        if suf[0] == "。":
            morphemes.append(neko_list)
            neko_list = []

import itertools
import matplotlib.pyplot as plt
import japanize_matplotlib
import collections

related_list = []
for sentense in morphemes:
    for morph in sentense:
        if morph['surface'] == '猫':
            for i in range(len(sentense)):
                if sentense[i]['surface']!='猫' and sentense[i]["pos"] != "補助記号" and sentense[i]["pos"] != '助詞' and sentense[i]["pos"] != '助動詞':
                    related_list.append(sentense[i]['surface'])
count_list = collections.Counter(related_list)
word_list = []
height_list = []

for i in range(10):
    word_list.append(count_list.most_common()[:10][i][0])
    height_list.append(count_list.most_common()[:10][i][1])
plt.bar(x = word_list, height = height_list)
plt.xlabel('word')
plt.ylabel('count')
plt.title('word_list_top10')
plt.savefig('related_word_list_top10.png')