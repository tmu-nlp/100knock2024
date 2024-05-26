from knock30 import sentence_list

noun_list =[]

for sentence in sentence_list:
    noun = ""
    count = 0
    for morph in sentence:
        if morph['pos']=='名詞':
            noun += morph['surface']
            count = count + 1
        else:
            if count > 1: #連接になっていないもの（単体の名詞）は省く
                noun_list.append(noun)
                noun = ""
                count = 0

if __name__ == '__main__':
    print(noun_list[:10])