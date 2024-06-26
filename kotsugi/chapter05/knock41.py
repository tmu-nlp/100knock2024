import re
from knock40 import Morph

class Chank:
  def __init__(self, morphs: list[Morph], dst: int, srcs: int):
    self.morphs = morphs
    self.dst = dst
    self.srcs = srcs
  
  def marge_morphs(self) -> str:
    return ''.join([ morph.surface for morph in self.morphs if morph.pos != '記号' ])
  
  def marge_morphs_only_nouns(self) -> str:
    return ''.join([ morph.surface for morph in self.morphs if morph.pos == '名詞' ])
  
  def include_pos(self, pos: str, pos1: str = None):
    for morph in self.morphs:
      if pos1 == None and morph.pos == pos:
        return True
      elif morph.pos == pos and morph.pos1 == pos1:
        return True
    return False

  def search_morph_by_pos(self, pos: str, pos1: str = None):
    if pos1 == None:
      return [morph for morph in self.morphs if morph.pos == pos]
    else :
      return [morph for morph in self.morphs if morph.pos == pos and morph.pos1 == pos1]
    
  def aliased(self, aliase: str) -> str:
    word = self.marge_morphs()
    nouns = self.marge_morphs_only_nouns()
    return word.replace(nouns, aliase)
  
def get_chanks() -> list[Chank]:
  chanks = []
  morphs = []

  with open('./kotsugi/chapter05/ai.ja.txt.parsed', 'r') as f:
    for line in f:
      if "EOS" in line:
        chanks.append(Chank(morphs, pre_dst, pre_srcs))
        morphs = []
        continue

      if m := re.match(r'\* (\d+?) (-?\d+?)D.+', line):
        dst = int(m.group(2))
        srcs = int(m.group(1))

        if len(morphs) > 0:
          chanks.append(Chank(morphs, pre_dst, pre_srcs))
          morphs = []
          pre_dst = dst
          pre_srcs = srcs
        else:
          pre_dst = dst
          pre_srcs = srcs
        continue

      lines = line.split('\t')
      surface = lines[0]
      features = lines[1].split(',')
      base = ''
      pos = features[0]
      pos1 = features[1]

      if len(features) >= 7:
        base = features[6]
      
      morphs.append(Morph(surface, base, pos, pos1))

  return chanks

def get_chanks_by_sentence() -> list[list[Chank]]: 
  sentences = []
  chanks = []
  morphs = []

  with open('./kotsugi/chapter05/ai.ja.txt.parsed', 'r') as f:
    for line in f:
      if "EOS" in line:
        chanks.append(Chank(morphs, pre_dst, pre_srcs))
        sentences.append(chanks)
        chanks = []
        morphs = []
        continue

      if m := re.match(r'\* (\d+?) (-?\d+?)D.+', line):
        dst = int(m.group(2))
        srcs = int(m.group(1))

        if len(morphs) > 0:
          chanks.append(Chank(morphs, pre_dst, pre_srcs))
          morphs = []
          pre_dst = dst
          pre_srcs = srcs
        else:
          pre_dst = dst
          pre_srcs = srcs
        continue

      lines = line.split('\t')
      surface = lines[0]
      features = lines[1].split(',')
      base = ''
      pos = features[0]
      pos1 = features[1]

      if len(features) >= 7:
        base = features[6]
      
      morphs.append(Morph(surface, base, pos, pos1))

  return sentences

if __name__ == "__main__":
  # chanks = get_chanks()

  # for chank in chanks[:10]:
  #   print(f"{chank.srcs}: chank: {chank.marge_morphs()} to: {chank.dst}")
    
  sentences = get_chanks_by_sentence()
  for i, chanks in enumerate(sentences):
    print(f"------index {i}------")
    for chank in chanks:
      print(f"{chank.srcs}: chank: {chank.marge_morphs()} to: {chank.dst}")