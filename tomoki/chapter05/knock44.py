#No44(係り受け木の可視化)

#graphivizを利用するためのコマンド　sudo apt install graphviz
import pydot_ng as pydot
pair = []

class Sentence:
    #コンストラクタを定義する
    #chunksはChunkクラスのインスタンスのリスト
  def __init__(self, chunks):
    #インスタンス変数に値を受け取らせる
    self.chunks = chunks
    #self.chunksの中身とインデックス番号をchunkとiに入れていく
    for i, chunk in enumerate(self.chunks):
      #chunk.dstがNoneも-1も含まない場合、srcsリストにインデックス番号を追加する
      if chunk.dst not in [None, -1]:
        self.chunks[chunk.dst].srcs.append(i)
 
class Chunk:
  #コンストラクタを定義する
  def __init__(self, morphs, dst, chunk_id):
    #インスタンス変数に値を受け取らせる
    self.morphs = morphs
    self.dst = dst
    self.srcs = []
    self.chunk_id = chunk_id

class Morph:
  #コンストラクタを定義する
  def __init__(self, line):
    #lineをsurfaceとotherに分離する
    surface, other = line.split("\t")
    #otherを,で分けていく
    other = other.split(",")
    #インスタンス変数に値を受け取らせる
    self.surface = surface
    self.base = other[-3]
    self.pos = other[0]
    self.pos1 = other[1]
 
sentences = [] #文リスト
chunks = [] #節リスト
morphs = [] #形態素リスト
chunk_id = 0 #文節番号
 
with open("ai.ja.txt.parsed") as f:
  #ファイルを一行ずつ読み込む
  for line in f:
    #アスタリスクは文節の開始位置を意味する
    if line[0] == "*":
      #morphsリストに何か入っている場合、Chunkクラスを呼び出し、それをchunksリストに追加
      #その後、chunk_idを1増やし、morphsリストをリセットする
      if morphs:
        chunks.append(Chunk(morphs, dst, chunk_id))
        chunk_id += 1
        morphs = []
      dst = int(line.split()[2].replace("D", ""))
    #文節の開始位置でも文末でもない場合、Morohクラスを呼び出し、それをmorphsリストに追加
    elif line != "EOS\n":
      morphs.append(Morph(line))
    #上記の2つの条件に当てはまらない場合、Chunkクラスを呼び出し、それをchunksリストに追加
    #その後、chunksを引数としてSentenceクラスを呼び出し、それをsentencesリストに追加し、それぞれのリストをリセットする
    else:
      chunks.append(Chunk(morphs, dst, chunk_id))#(list, int, int)
      sentences.append(Sentence(chunks))
      morphs = []
      chunks = []
      dst = None
      chunk_id = 0
#NO42とほとんど同じ処理（pair.appendの部分が異なる）
pair = []
for chunk in sentences[2].chunks:
  if int(chunk.dst) == -1:
    continue
  else:
    surf = "".join([morph.surface for morph in chunk.morphs if morph.pos != "記号"])
    next_surf = "".join([morph.surface for morph in sentences[2].chunks[int(chunk.dst)].morphs if morph.pos != "記号"]) #文節のリストに係り先番号をindexに指定。その文節の形態素リストを取得
    pair.append((surf, next_surf))
#pydot.Dot()でグラフを作成
img = pydot.Dot()
#minchoフォントを指定   
img.set_node_defaults(fontname="MS Mincho")
for s, t in pair:
  #有向グラフにエッジを追加する。
  img.add_edge(pydot.Edge(s, t))
#有向グラフpngファイルに保存
img.write_png("./result44.png")