#46.動詞の格フレーム情報の抽出
"""
45のプログラムを改変し，述語と格パターンに続けて項
（述語に係っている文節そのもの）をタブ区切り形式で出力せよ．
45の仕様に加えて，以下の仕様を満たすようにせよ．

項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる
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

with open("./result46.txt", "w") as f:
  for i in range(len(sentences)):
    for chunk in sentences[i].chunks:
      for morph in chunk.morphs:
        if morph.pos == "動詞": 
          particles = []
          items = []
          for src in chunk.srcs:
            particles += [morph.surface for morph in sentences[i].chunks[src].morphs if morph.pos == "助詞"]
            items += ["".join([morph.surface for morph in sentences[i].chunks[src].morphs if morph.pos != "記号"])]
          if len(particles) > 1:
            if len(items) > 1:
              particles = sorted(set(particles))
              items = sorted(set(items))
              particles_form = " ".join(particles)
              items_form = " ".join(items)
              print(f"{morph.base}\t{particles_form}\t{items_form}", file=f)
