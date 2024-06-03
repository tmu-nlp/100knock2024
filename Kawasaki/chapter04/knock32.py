import knock30

verb_base = []

for sentence in knock30.sentence_list:
    for morph in sentence:
        if morph['pos'] == "動詞":
            verb_base.append(morph['base'])

if __name__ == '__main__':
    print(verb_base[:10])