class Morph:
    def __init__(self, words):
        surface, words1 = words.split('\t')
        words1 = words1.split(',')
        self.surface = surface
        self.base = words1[6]
        self.pos = words1[0]
        self.pos1 = words1[1]


# クラスChunkでは、形態素リスト、係り先文節のインデックス番号、係り元文節のインデックス番号リストを保持
class Chunk():
    def __init__(self, morphs, a, chunk_id):
        self.morphs = morphs
        self.dst = a  # 係り先文節index番号
        self.srcs = []  # 係り元文節index番号のリスト

# クラスSentenceは、文節のリストを保持し、各文節が係っている文節のインデックス番号を指定
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
        if line[0] == "*":  # 行が"*"で始まる場合、新しい文節の開始を示しています。この場合、文節の係り先の情報が含まれています。
            if morphs:  # 空ではないなら、すでに別の文節の形態素が読み込まれているため、それらの形態素をChunkオブジェクトにまとめ
                chunks.append(Chunk(morphs, dst, chunk_id))
                chunk_id += 1
                morphs = []
            dst = int(line.split()[2].replace("D", ""))  # 係先の番号
        elif line != "EOS\n":  # 行が"EOS\n"でない場合、形態素を処理します。形態素は各行がタブで区切られたテキストとして表されています。
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

