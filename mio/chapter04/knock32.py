#32. 動詞の基本形
#動詞の基本形をすべて抽出せよ．

import knock30


baseofverb = set()

for sentence in knock30.morph_results:
  for word in sentence:
    if word["pos"] == "動詞":
      baseofverb.add(word["base"])

print(baseofverb)

