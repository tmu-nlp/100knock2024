class Morph():
    def __init__(self, morph):
        surface, attr = morph.split('\t')
        attr_list = attr.split(',')
        self.surface = surface
        self.base = attr_list[6]
        self.pos = attr_list[0]
        self.pos1 = attr_list[1]

class Chunk():
    #文節を表すオブジェクトを定義
    def __init__(self,morphs,dst):
        #文節の形態素リスト
        self.morphs = morphs
        #かかり先の文節のインデックス
        self.dst = dst 
        #かかり元の文節のリスト
        self.srcs = []

class Sentence():
    def __init__(self,chunks):
        self.chunks = chunks #チャンクのリスト
        for i, chunk in enumerate(self.chunks):
            if chunk.dst != -1: #係受け先が存在
                self.chunks[chunk.dst].srcs.append(i)
                #chunks[chunk.dst]はかかり受け先のindex
                #.srcsはかかり元文節インデックス番号のリスト
                """ 
                例) 助詞 ... 動詞
                  (index4)  (index8)
                 ⇒self.chunks[chunk.dst] ⇒ 8
                   self.chunks[chunk.dst].srcs ⇒ 元々のかかり元キー
                   self.chunks[chunk.dst].srcs.append(i) ⇒4追加
                 """

sentences = []
morphs = []
chunks = []

with open('ai.ja.txt.parsed','r') as f:
    for line in f:
        if line[0] == '*':
            if len(morphs) > 0:
                chunks.append(Chunk(morphs,dst))
                morphs = []
            dst = int(line.split(' ')[2].rstrip('D'))
        elif line == 'EOS\n':
            if len(morphs) > 0:
                chunks.append(Chunk(morphs,dst))
                sentences.append(Sentence(chunks))
            morphs = []
            chunks = []
            dst = None
        else :
            morphs.append(Morph(line))
print(sentences[1])

# for chunk in sentences[1].chunks:
#     print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)

