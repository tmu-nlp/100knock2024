import re
import numpy as np

path = 'neko.txt.mecab'
with open(path) as f:
    text = f.read().split('\n')

sentence_list = list() #文ごとのリストを全て含むリストを用意
dict_line = dict() #形態素をいれる辞書を用意
sentence = list() #一文を入れるリストを用意

for line in text:
    split_line = re.split('[\t,]',line)
    if len(split_line) == 1 and split_line[0] == "": #何もない時はスキップ
        continue
    elif len(split_line) == 1 and split_line[0] == "EOS": #End Of Statement
        sentence_list.append(sentence) #一文の形態素が入ったリストをsentence_listに追加
        sentence = list() #一文を入れるリストを用意
        continue
    dict_line["surface"] = split_line[0]
    dict_line["base"] = split_line[7]
    dict_line["pos"] = split_line[1]
    dict_line["pos1"] = split_line[2]
    sentence.append(dict_line) #形態素の辞書をsentenceについか
    dict_line = dict() #形態素をいれる辞書を用意

if __name__ == '__main__':
    print(sentence_list[:10])
    # aaaaaa