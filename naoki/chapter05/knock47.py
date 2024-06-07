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
        for morph in chunk.morphs:
            if morph.pos == '動詞':
                part = []
                part1 = []
                part2 = []
                #かかり元チェック
                for i, src in enumerate(chunk.srcs):
                    if len(sentence.chunks[src].morphs) ==2 and sentence.chunks[src].morphs[0].pos1=='サ変接続' and sentence.chunks[src].morphs[1].surface=='を':
                        verb = sentence.chunks[src].morphs[0].surface + sentence.chunks[src].morphs[1].surface + morph.base
                        part.append(verb)
                        for src in chunk.srcs:
                            for morph2 in sentence.chunks[src].morphs:
                                noun_particle = ''
                                if morph2.pos == '名詞':    
                                    noun_particle += morph2.surface
                                    for morph3 in sentence.chunks[src].morphs:
                                        if morph3.pos == '助詞' :
                                            part1.append(morph3.surface)
                                            noun_particle += morph3.surface
                                            if noun_particle:
                                                part2.append(noun_particle)
                if len(part2) > 0:
                    sort_part1 = sorted(list(set(part1)))
                    sort_part2 = sorted(list(set(part2)))
                    part_line2 = ' '.join(part2)
                    part_line1 = ' '.join(part1)
                    part_line = ' '.join(part)
                    line = part_line + '\t' + part_line1 + '\t' + part_line2
                    print(line)
                    break

