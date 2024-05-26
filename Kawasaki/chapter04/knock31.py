from knock30 import sentence_list

verb_surface = []

for sentence in sentence_list:
    for morph in sentence:
        if morph['pos'] == "動詞":
            verb_surface.append(morph['surface'])

if __name__ == '__main__':
    print(verb_surface[:10])