def parse_neko(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        sentences = [] # 1文ごとの形態素リストのリスト
        morphs = [] # 形態素の辞書を格納
        for line in f:
            if line == 'EOS\n':
                if len(morphs) > 0:
                    sentences.append(morphs)
                    morphs = []
            else:
                surface, feature = line.split('\t')
                features = feature.split(',')
                base = features[6] if features[6] != '*' else surface
                pos, pos1 = features[0], features[1]
                morph = {
                    'surface': surface,
                    'base': base,
                    'pos': pos,
                    'pos1': pos1
                }
                morphs.append(morph)
        return sentences 

if __name__ == "__main__":
    sentences = parse_neko('neko.txt.mecab')
    print(sentences[:2])  # 最初の2文を表示