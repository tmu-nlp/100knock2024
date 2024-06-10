"""
#42.係り元と係り先の文節の表示
係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
ただし，句読点などの記号は出力しないようにせよ
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
    if int(chunk.dst) == -1:
      continue
    else:
      surf = "".join([morph.surface for morph in chunk.morphs if morph.pos != "記号"])
      next_surf = "".join([morph.surface for morph in sentences[2].chunks[int(chunk.dst)].morphs if morph.pos != "記号"]) 
      print(f"{surf}\t{next_surf}")
    