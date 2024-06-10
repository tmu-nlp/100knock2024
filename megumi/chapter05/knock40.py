#40
"""
形態素を表すクラスMorphを実装せよ．
このクラスは表層形（surface），基本形（base），
品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．
さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，
各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ．
"""

class Morph:
  def __init__(self, line):
    surface, other = line.split("\t")
    other = other.split(",")
    self.surface = surface #表層形
    self.base = other[6]   #基本形
    self.pos = other[0]    #品詞
    self.pos1 = other[1]   #品詞再分類１

sentences = [] #文リスト
morphs = []    #形態素リスト

with open("./ai.ja.txt.parsed") as f:
  for line in f:
    if line[0] == "*":
      continue
    elif line != "EOS\n": 
      morphs.append(Morph(line))
    else:  #EOS（文末）の場合
        if len(morphs) !=0:
           sentences.append(morphs)
           morphs = []

for i in sentences[1]:
    print(vars(i))
