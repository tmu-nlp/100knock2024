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

import collections
word_list = []
for sentense in morphemes:
    for morph in sentense:
        if morph['pos'] != '補助記号' and morph['pos'] != '助詞' and morph['pos'] != '助動詞':
            word_list.append(morph['surface'])
word_list_count = collections.Counter(word_list)
word_list_rank = word_list_count.most_common()

import matplotlib.pyplot as plt
import japanize_matplotlib
import collections

word_list_top10 = []
word_list_count = []
for i in range(10):
    word_list_top10.append(word_list_rank[:10][i][0])
    word_list_count.append(word_list_rank[:10][i][1])
plt.bar(x = word_list_top10,height = word_list_count)
plt.xlabel('word')
plt.ylabel('count')
plt.title('word_list_top10')
plt.savefig('word_list_top10.png')
