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

word_frequency_list = []
for sentense in morphemes:
    for morph in sentense:
        word_frequency_list.append(morph['surface'])
plot = collections.Counter(word_frequency_list)
#プロットなので降順に並び変える必要がある
plot_rank = sorted((plot.values()), reverse = True)
plt.plot(plot_rank)
plt.xlabel('freqency')
plt.ylabel('count')
plt.title('zipf')
#現在のaxesオブジェクト(軸を含んだグラフ)を渡す
ax = plt.gca()
ax.set_yscale('log')  # メイン: y軸をlogスケールで描く
ax.set_xscale('log')
plt.savefig('zipf.png')
