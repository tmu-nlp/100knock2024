class Morph:
    def __init__(self, words):
        surface, words1 = words.split('\t')
        words1 = words1.split(',')
        self.surface = surface
        self.base = words1[6]
        self.pos = words1[0]
        self.pos1 = words1[1]


class Chunk():
    def __init__(self, morphs, a, chunk_id):
        self.morphs = morphs
        self.dst = a  # 係り先文節インデックス番号
        self.srcs = []  # 係り元文節インデックス番号のリスト


class Sentence():  # 係り先文節インデックス番号のため
    def __init__(self, chunks):
        self.chunks = chunks
        for i, chunk in enumerate(self.chunks):  # enumerateの輸出：(index,内容)
            if chunk.dst not in [None, -1]:
                self.chunks[chunk.dst].srcs.append(
                    i)  # accesses the chunk at chunk.dst in the self.chunks list, appends the value of i to the srcs.


sentences = []  # 文リスト
chunks = []  # 節リスト
morphs = []  # 形態素リスト
chunk_id = 0  # 文節番号

with open("ai.ja.txt.parsed", encoding='UTF-8') as f:
    for line in f:
        if line[0] == "*":
            if morphs:  # 空ではないなら
                chunks.append(Chunk(morphs, dst, chunk_id))
                chunk_id += 1
                morphs = []
            dst = int(line.split()[2].replace("D", ""))  # 係先の番号
        elif line != "EOS\n":
            morphs.append(Morph(line))
        else:  # EOS\n
            chunks.append(Chunk(morphs, dst, chunk_id))
            sentences.append(Sentence(chunks))
            morphs = []
            chunks = []
            dst = None
            chunk_id = 0

for chunk in sentences[2].chunks:  # 第一の句
    chunk_str = "".join([morph.surface for morph in chunk.morphs])  # surface
    print(f"{chunk_str} {chunk.dst} \n")

