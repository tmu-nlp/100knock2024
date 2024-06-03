"""
#45.動詞の格パターンの抽出
今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい． 
動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ． 
ただし，出力は以下の仕様を満たすようにせよ．

動詞を含む文節において，最左の動詞の基本形を述語とする
述語に係る助詞を格とする
述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
このプログラムの出力をファイルに保存し，以下の事項をUNIXコマンドを用いて確認せよ．

コーパス中で頻出する述語と格パターンの組み合わせ
「行う」「なる」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）
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
      
with open("./result45.txt", "w") as f:
  for i in range(len(sentences)):
    for chunk in sentences[i].chunks:
      for morph in chunk.morphs:
        if morph.pos == "動詞": 
          particles = []
          for src in chunk.srcs:
            particles += [morph.surface for morph in sentences[i].chunks[src].morphs if morph.pos == "助詞"]
          if len(particles) > 1:
            particles = set(particles)
            particles = sorted(list(particles))
            form = " ".join(particles)
            print(f"{morph.base}\t{form}", file=f)