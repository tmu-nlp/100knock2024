#41.係り受け解析結果の読み込み（文節・係り受け）

"""
40に加えて，文節を表すクラスChunkを実装せよ．
このクラスは形態素（Morphオブジェクト）のリスト（morphs），係り先文節インデックス番号（dst），
係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする．
さらに，入力テキストの係り受け解析結果を読み込み，
１文をChunkオブジェクトのリストとして表現し，冒頭の説明文の文節の文字列と係り先を表示せよ．
本章の残りの問題では，ここで作ったプログラムを活用せよ
"""

class Sentence:
  def __init__(self, chunks):
    self.chunks = chunks
    for i, chunk in enumerate(self.chunks):
      if chunk.dst not in [None, -1]:
        self.chunks[chunk.dst].srcs.append(i)

class Chunk:
  def __init__(self, morphs, dst, chunk_id):
    self.morphs = morphs
    self.dst = dst
    self.srcs = []
    self.chunk_id = chunk_id

class Morph:
  def __init__(self, line):
    surface, other = line.split("\t")
    other = other.split(",")
    self.surface = surface
    self.base = other[6]
    self.pos = other[0]
    self.pos1 = other[1]

sentences = [] #文リスト
chunks = []    #節リスト
morphs = []    #形態素リスト
chunk_id = 0   #文節番号

with open("./ai.ja.txt.parsed") as f:
  for line in f:
    if line[0] == "*":
      if morphs:
        chunks.append(Chunk(morphs, dst, chunk_id))
        chunk_id += 1
        morphs = []
      dst = int(line.split()[2].replace("D", ""))
    elif line != "EOS\n":
      morphs.append(Morph(line))
    else:
      chunks.append(Chunk(morphs, dst, chunk_id))
      sentences.append(Sentence(chunks))
 
      morphs = []
      chunks = []
      dst = None
      chunk_id = 0

for chunk in sentences[2].chunks:
  chunk_str = "".join([morph.surface for morph in chunk.morphs])
  print(f"文節の文字列：{chunk_str}\n係り先の文節番号：{chunk.dst}\n")

