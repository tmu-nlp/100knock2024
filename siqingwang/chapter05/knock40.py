class Morph:
    def __init__(self, words):
        surface, words1 = words.split('\t')
        words1 = words1.split(',')
        self.surface = surface
        self.base = words1[6]
        self.pos = words1[0]
        self.pos1 = words1[1]


sentences = []
morphs = []

with open('ai.ja.txt.parsed', encoding='UTF-8') as f:
    for line in f:
        if line[0] == '*':
            continue
        elif line != 'EOS\n':
            morphs.append(Morph(line))
        else:  # EOS\n
            sentences.append(morphs)
            morphs = []

for i in sentences[2]:  # 第一句
    print(vars(i))  # vars(): オブジェクトの属性とそれに対応する値を取得する
