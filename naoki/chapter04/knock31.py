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
    verb_list = []
    for sentence in morphemes:
        for morph in sentence:
            if morph['pos'] == '動詞':
                verb_list.append(morph['surface'])
print(verb_list)