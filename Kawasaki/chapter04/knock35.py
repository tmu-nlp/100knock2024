from knock30 import sentence_list
from collections import defaultdict #存在チェックが不要
from collections import OrderedDict #順番ありの辞書にする

word_dict = defaultdict(lambda: 0) #lambdaは初期化の関数

for sentence in sentence_list:
    for morph in sentence:
        if morph["pos"] == "記号": #記号はカウントしない
            continue
        word_dict[morph["base"]] += 1

sort_word_dict = OrderedDict(sorted(word_dict.items(), key = lambda x:x[1], reverse=True)) 

if __name__ == '__main__':
    for k,v in sort_word_dict.items():
        print(f"{k} : {v}")