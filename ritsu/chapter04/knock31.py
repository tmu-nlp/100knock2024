from knock30 import parse_neko

def extract_verbs(sentences):
    verbs = []
    for sentence in sentences:
        for morph in sentence:
            if morph['pos'] == '動詞':
                verbs.append(morph['surface'])
    return verbs

if __name__ == "__main__":
    sentences = parse_neko('neko.txt.mecab')
    verbs = extract_verbs(sentences)
    print(verbs[:10])  # 最初の10個の動詞を表示