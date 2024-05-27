import re
import numpy as np

with open('neko2.txt.mecab','r') as f:
    neko_data = f.read()
split_neko = neko_data.split("\n")

sentence_list = list()
dict_line = dict()
sentence = list()

for line in split_neko:
    split_line = re.split('[\t,]',line)
    #print(split_line)
    if len(split_line) == 1 and split_line[0] == "":
        continue
    if len(split_line) == 1 and split_line[0] == "EOS":
        sentence_list.append(sentence)
        sentence = list()
        continue
    #print(split_line[0],split_line[7],split_line[1],split_line[2])
    dict_line["surface"] = split_line[0]
    dict_line["base"] = split_line[7]
    dict_line["pos"] = split_line[1]
    dict_line["pos1"] = split_line[2]
    sentence.append(dict_line)
    dict_line = dict()