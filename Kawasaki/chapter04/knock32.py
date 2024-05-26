from knock30 import sentence_list

verb_base = []

for sentence in sentence_list:
    for morph in sentence:
        if morph['pos'] == "動詞":
            verb_base.append(morph['base'])

if __name__ == '__main__':
    print(verb_base[:10])