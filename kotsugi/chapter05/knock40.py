import re

class Morph:
  def __init__(self, surface: str, base: str, pos: str, pos1: str):
    self.surface = surface
    self.base = base
    self.pos = pos
    self.pos1 = pos1

def get_morphs(): 
  sentences = []
  with open('./kotsugi/chapter05/ai.ja.txt.parsed', 'r') as f:
    morphs = []
    for line in f:
      if ("EOS" in line):
        sentences.append(morphs)
        morphs = []
        continue

      if (re.match(r'\*.+', line)):
        continue

      lines = line.split('\t')
      surface = lines[0]
      features = lines[1].split(',')
      base = ''
      pos = features[0]
      pos1 = features[1]

      if (len(features) >= 7):
        base = features[6]
      
      morphs.append(Morph(surface, base, pos, pos1))

  return sentences

if __name__ == "__main__":
  sentences = get_morphs()

  morphs = sentences[2]
  for morph in morphs:
    print('---------')
    print(f"\tsurfase: {morph.surface}")
    print(f"\tbase   : {morph.base}")
    print(f"\tpos    : {morph.pos}")
    print(f"\tpos1   : {morph.pos1}")

# 係り受けファイル生成には'sed -e 's/。/。\n/g' ai.ja.txt | cabocha -f1 > ai.ja.txt.parsed'を実行