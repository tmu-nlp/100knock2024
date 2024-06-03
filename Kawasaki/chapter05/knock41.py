#形態素を表すクラスMorph

class Morph():
    def __init__(self, morph):
        surface, attr = morph.split('\t')
        attr_list = attr.split(',')
        self.surface = surface
        self.base = attr_list[6]
        self.pos = attr_list[0]
        self.pos1 = attr_list[1]

#形態素のチャンクを表すクラスChunk

class Chunk():
    def __init__(self, morphs, dst):
        self.morphs = morphs  # 形態素のリスト
        self.dst = dst  # 係り先文節インデックス番号
        self.srcs = []  # 係り元文節インデックス番号のリスト

#1つの文の全てのチャンクを表すSentence

class Sentence():
    def __init__(self, chunks):
        self.chunks = chunks  # Chunk型のリスト ここに一文の全てのchunkが揃っている
        for i, chunk in enumerate(self.chunks):  # 番号が欲しいためenumerateでfor文
            if chunk.dst != -1:  # 係り先が存在する場合
                self.chunks[chunk.dst].srcs.append(i) #係り先チャンクの係り元にiを追加
                # (self.chunks[chunk.dst].srcs)にiを追加
                # chunks[chunk.dst]は係り受け先のindex
                # .srcsは係り元文節インデックス番号のリスト

#テキストの係り受け解析表現

sentences = []
morphs = []
chunks = []

with open("ai.ja.txt.parsed", "r") as f:
    for line in f:
        if line[0] == '*': #塊の区切り目
            if len(morphs) > 0: #morphsにそれまでのMorphクラスのオブジェクトがあるなら
                chunks.append(Chunk(morphs, dst))  #Chunkクラスに変えてchunksに追加
                morphs = []  # 初期化
            dst = int(line.split(' ')[2].rstrip('D'))  # 係り先を取得
        elif line == "EOS\n": #1つの文の区切れ目
            if len(morphs) > 0:  # morphの中身がある場合
                chunks.append(Chunk(morphs, dst))  # chunksにchunkオブジェクトを追加
                sentences.append(Sentence(chunks))  # Sentenceオブジェクトに変えてsentencesにappend
            morphs = []
            chunks = []
            dst = None
        else:
            morphs.append(Morph(line))  # *でもEOSでもない場合、Morphオブジェクトに加えてmorphsにためていく

if __name__ == '__main__':
    for chunk in sentences[1].chunks:
        print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)