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

with open('verb_par.txt',mode='w') as f1:
    for sentence in sentences:
        for chunk in sentence.chunks:
            for morph1 in chunk.morphs:
                if morph1.pos == '動詞':
                    part = []
                    #かかり元チェック
                    for src in chunk.srcs:
                        for morph2 in sentence.chunks[src].morphs:
                            if morph2.pos == '助詞':    
                                part.append(morph2.surface)
                    #チェックリスト
                    if len(part) > 0:
                        #重複を削除した集合⇒リスト化⇒ソート
                        sort_part = sorted(list(set(part)))
                        part_line = ' '.join(part)
                        line = morph1.base + '\t' + part_line
                        print(line)
                        f1.write(line+'\n') 
                    break #動詞が見つかればループを抜け出す
"""
unixコマンド
sort verb_par.txt|uniq -c |sort -nr|grep -e "行う" -e "なる" -e"与える"
"""







