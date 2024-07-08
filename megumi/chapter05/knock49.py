#49.名詞間の係り受けパスの抽出
"""
文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ．
ただし，名詞句ペアの文節番号がiとj（i<j）のとき，係り受けパスは以下の仕様を満たすものとする．

問題48と同様に，パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を” -> “で連結して表現する
文節iとjに含まれる名詞句はそれぞれ，XとYに置換する
また，係り受けパスの形状は，以下の2通りが考えられる．

文節iから構文木の根に至る経路上に文節jが存在する場合
　: 文節iから文節jのパスを表示
上記以外で，文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合
　: 文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，文節kの内容を” | “で連結して表示
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

from itertools import combinations
import re

sentence = sentences[2]
nouns = []

with open("./result49.txt", "w") as f:
    # 文節を反復させ、名詞を含む文節のインデックス番号をnounsに格納する。
    for i, chunk in enumerate(sentence.chunks):
        if [morph for morph in chunk.morphs if morph.pos == "名詞"]:
            nouns.append(i)
    
    # 名詞句ペアの文節番号i、jを反復させる。
    for i, j in combinations(nouns, 2):
        path_I = []
        path_J = []
        
        # 文節iの構文木根に至る経路上に文節jが存在する場合とそれ以外で条件分岐を行う
        while i != j:
            if i < j:  # 文節iの構文木経路上に文節jが存在する場合
                path_I.append(i)
                i = sentence.chunks[i].dst
            else:  # 文節iの構文木経路上に文節jがない場合
                path_J.append(j)
                j = sentence.chunks[j].dst
        
        if len(path_J) == 0:  # 文節Iの構文木上に文節Jが存在する場合
            # 文節iとjに含まれる名詞句をXとYに置換する。
            X = "X" + "".join([morph.surface for morph in sentence.chunks[path_I[0]].morphs if morph.pos != "名詞" and morph.pos != "記号"]) 
            Y = "Y" +  "".join([morph.surface for morph in sentence.chunks[i].morphs if morph.pos != "名詞" and morph.pos != "記号"])
            chunk_X = re.sub("X+", "X", X)
            chunk_Y = re.sub("Y+", "Y", Y)
            # 残りは、文節iと文節jの構文木経路の間のパスを取得し、連結させて出力する。
            path_ItoJ = [chunk_X] + ["".join(morph.surface for n in path_I[1:] for morph in sentence.chunks[n].morphs)] + [chunk_Y]
            f.write(" -> ".join(path_ItoJ) + "\n")
        else:  # 文節Iの構文木上に文節Jが存在しない場合
            X = "X" + "".join([morph.surface for morph in sentence.chunks[path_I[0]].morphs if morph.pos != "名詞" and morph.pos != "記号"]) 
            Y = "Y" + "".join([morph.surface for morph in sentence.chunks[path_J[0]].morphs if morph.pos != "名詞" and morph.pos != "記号"]) 
            chunk_X = re.sub("X+", "X", X)
            chunk_Y = re.sub("Y+", "Y", Y)
            chunk_k = "".join([morph.surface for morph in sentence.chunks[i].morphs if morph.pos != "記号"])
            path_X = [chunk_X] + ["".join(morph.surface for n in path_I[1:] for morph in sentence.chunks[n].morphs if morph.pos != "記号")]
            path_Y = [chunk_Y] + ["".join(morph.surface for n in path_J[1:] for morph in sentence.chunks[n].morphs if morph.pos != "記号")]
            f.write(" | ".join(["->".join(path_X), "->".join(path_Y), chunk_k]) + "\n")

