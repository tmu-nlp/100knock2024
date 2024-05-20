from knock30 import parse_neko

def extract_verb_bases(sentences):
    verb_bases = []
    for sentence in sentences:
        for morph in sentence:
            if morph['pos'] == '動詞':
                verb_bases.append(morph['base'])
    return verb_bases

if __name__ == "__main__":
    sentences = parse_neko('neko.txt.mecab')
    verb_bases = extract_verb_bases(sentences)
    print(verb_bases[:10])  # 最初の10個の動詞の基本形を表示