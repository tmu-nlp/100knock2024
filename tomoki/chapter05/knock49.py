#NO49(名詞間の係り受けパスの抽出)
#かなり難しいので、要復習
class Sentence:
    #コンストラクタを定義する
  def __init__(self, chunks):
    #インスタンス変数に値を受け取らせる
    self.chunks = chunks
    #self.chunksの中身とインデックス番号をchunkとiに入れていく
    for i, chunk in enumerate(self.chunks):
      #chunk.dstにNoneも-1も含まれない場合、srcsリストにインデックス番号を追加する
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
    #その後、chanksを引数としてSentenceクラスを呼び出し、それをsentencesリストに追加し、それぞれのリストをリセットする
    else:
      chunks.append(Chunk(morphs, dst, chunk_id))
      sentences.append(Sentence(chunks))
      morphs = []
      chunks = []
      dst = None
      chunk_id = 0

from itertools import combinations
import re
#名詞(nouns)を探して、名詞を含む文節のインデックス番号をnounsに追加する。
sentence = sentences[2]
nouns = []
for i, chunk in enumerate(sentence.chunks):
  if [morph for morph in chunk.morphs if morph.pos == "名詞"]:
    nouns.append(i)
#組み合わせを出したいときは、はitertoolsのcombinations(組み込み関数)を利用する。
for i, j in combinations(nouns, 2):
  
  path_I = []
  path_J = []
  #i=jになるまで繰り返す
  while i != j:
    #文節iの構文木経路上に文節jが存在する場合、path_I(文節I)に文節番号iを追加する。
    if i < j: 
      path_I.append(i)
      i = sentence.chunks[i].dst
    #文節iの構文木経路上に文節jがない場合、path_J(文節J)に文節番号jを追加する。
    else: 
      path_J.append(j)
      j = sentence.chunks[j].dst
  # 文節Iの構文木上に文節Jが存在する場合
  if len(path_J) == 0: 
    # 名詞でも記号でもないものを、表層形として取得して、それを結合する。
    X = "X" + "".join([morph.surface for morph in sentence.chunks[path_I[0]].morphs if morph.pos != "名詞" and morph.pos != "記号"]) 
    Y = "Y" +  "".join([morph.surface for morph in sentence.chunks[i].morphs if morph.pos != "名詞" and morph.pos != "記号"])
    chunk_X = re.sub("X+", "X", X)
    chunk_Y = re.sub("Y+", "Y", Y)
    path_ItoJ = [chunk_X] + ["".join(morph.surface for n in path_I[1:] for morph in sentence.chunks[n].morphs)] + [chunk_Y]
    print(" -> ".join(path_ItoJ))
    # 文節Iの構文木上に文節Jが存在しない場合
  else: 
    # 名詞でも記号でもないものを、表層形として取得して、それを結合する。
    X = "X" + "".join([morph.surface for morph in sentence.chunks[path_I[0]].morphs if morph.pos != "名詞" and morph.pos != "記号"]) 
    Y = "Y" + "".join([morph.surface for morph in sentence.chunks[path_J[0]].morphs if morph.pos != "名詞" and morph.pos != "記号"]) 
    chunk_X = re.sub("X+", "X", X)
    chunk_Y = re.sub("Y+", "Y", Y)
    #文節iと文節jの構文木の根に至る経路上で交わる文節kを取得する。
    chunk_k = "".join([morph.surface for morph in sentence.chunks[i].morphs if morph.pos != "記号"])
    #文節iと文節jで、文節kに至るまでのパスを取得する
    path_X = [chunk_X] + ["".join(morph.surface for n in path_I[1:] for morph in sentence.chunks[n].morphs if morph.pos != "記号")]
    path_Y = [chunk_Y] + ["".join(morph.surface for n in path_J[1: ]for morph in sentence.chunks[n].morphs if morph.pos != "記号")]
    #文節iから文節kに至るまでのパスと、文節jから文節kに至るまでのパスと、文節kの内容をそれぞれ” | “で連結させて、出力する。
    print(" | ".join(["->".join(path_X), "->".join(path_Y), chunk_k]))