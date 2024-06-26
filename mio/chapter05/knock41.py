#41. 係り受け解析結果の読み込み（文節・係り受け）
#40に加えて，文節を表すクラスChunkを実装せよ．
# このクラスは形態素（Morphオブジェクト）のリスト（morphs），係り先文節インデックス番号（dst），係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
# さらに，入力テキストの係り受け解析結果を読み込み，１文をChunkオブジェクトのリストとして表現し，冒頭の説明文の文節の文字列と係り先を表示せよ．本章の残りの問題では，ここで作ったプログラムを活用せよ．

#参考：https://mori-memo.hateblo.jp/entry/2022/09/03/222705
class Morph:
    def __init__(self, morph):
        
        #➀morphemeをタブの前後で分割し、前半部分をsurface/後半部分をattributeに格納
        #morphemeの例）人工	名詞,一般,*,*,*,*,人工,ジンコウ,ジンコー
        surface, attribute = morph.split('\t')
        attribute  = attribute.split(",")
        self.surface = surface
        self.base = attribute[6] #人工
        self.pos = attribute[0] #名詞
        self.pos1 = attribute[1] #一般

#*で始まる行（Morphが取得しなかった行/以下スペース区切り）：文節番号　係り先の文節番号(係り先なし:-1)(語尾にD)　主辞の形態素番号/機能語の形態素番号　係り関係のスコア(大きい方が係りやすい)
#例）* 0 17D 1/1 -1.77692
#dstに係り先の文節番号を入れる
#srcs：空リストで初期化（係り元なので、一文すべてのChunkを取得し終えてからでないとわからない）
#      →一文すべてのChunkクラスのリストが完成した後、ひとつずつ探索
# 　   →係り先のChunkクラスのsrcsに係り元の文節番号をappend

class Chunk():
    def __init__(self, morphs, dst):
        self.morphs =  morphs
        self.dst = dst
        self.srcs = []

class Sentence:
    def __init__(self, chunks):
        self.chunks = chunks ##文中の全ての文節のリストを持つ SentenceクラスにはChunkクラスが入っている
        for index_chunks, chunk in enumerate(self.chunks):
            #係受け先が存在するとき、
            if chunk.dst != -1:
            #self.chunks_list[chunk.dst]：現在の文節chunkが係っている文節
            #文節インデックス番号index_chanksのsrcsリストに、係り先を格納するdstの番号に対応するChunkをのリストに格納   
                self.chunks[chunk.dst].srcs.append(index_chunks)
    
sentences = []
morphs = []
chunks = []

with open("ai.ja.txt.parsed", "r") as f:
    for line in f:
        if line[0] == '*':
            if len(morphs) > 0: #morphs_tmpが空でないとき
                chunks.append(Chunk(morphs, dst))  # chunkの区切り目になるのでappend
                morphs = []  # 初期化
            #文節の係り受け情報の行（Morphがスキップした行）から、係り先の文節インデックスを抽出
            #例：* 0 2D
            dst = int(line.split(' ')[2].rstrip('D'))  
        
        elif line == "EOS\n":
            if len(morphs) > 0:
                chunks.append(Chunk(morphs, dst))  # chunksにchunkオブジェクトを追加
                sentences.append(Sentence(chunks))  # sentenceの区切りめになるのでappend
                morphs = []
                chunks = []
                dst = None
        else:
            morphs.append(Morph(line))  # morphを追加
    
#for i in range(3):
for sentence in sentences:
    for chunk in sentence.chunks:
    #for chunk in sentences[i].chunks:
        surfaces = []
#chunk.morphsの各morphをループで取り出す→そのsurface属性をsurfacesリストに追加
#内包表記：print([morpheme.surface for morpheme in chunk.morphs_tmp], chunk.dst, chunk.srcs)
        """for morph in chunk.morphs:
            surfaces.append(morph.surface)
        print(surfaces, chunk.dst, chunk.srcs)"""
    

