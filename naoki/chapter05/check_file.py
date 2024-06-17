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

sentences = []
morphs = []
chunks = []

with open('ai.ja.txt.parsed','r') as f:
    for line in f:
        if line[0] == '*':
            if len(morphs) >0:
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

for sentence in sentences:
    for chunk in sentence.chunks:
        if chunk.dst != -1:
            modiin = []
            modifor = []
            normbool = 0
            verbbool = 0
            for morph in chunk.morphs:
                if morph.pos == '名詞':
                    modiin.append(morph.surface)
                    normbool = 1
            for morph in sentence.chunks[chunk.dst].morphs:
                if morph.pos == '動詞':
                    modifor.append(morph.surface)
                    verbbool = 1
            phrasein = ''.join(modiin)
            phraseout = ''.join(modifor)
            if normbool == 1 and verbbool ==1:
                print(f'{phrasein}\t{phraseout}')

