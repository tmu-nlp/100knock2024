import knock30

verb_surface = []

for sentence in knock30.sentence_list:
    for morph in sentence:
        if morph['pos'] == "動詞":
            verb_surface.append(morph['surface'])

if __name__ == '__main__':
    print(verb_surface[:10])