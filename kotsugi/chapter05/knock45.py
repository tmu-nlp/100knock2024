from knock40 import Morph
from knock41 import get_chanks_by_sentence, Chank

def search_postpos_morphs(chanks: list[Chank], dst: int) -> list[Morph]:
  postposies = []
  for chank in chanks:
    if chank.dst == dst:
      postpos = chank.search_morph_by_pos("助詞")

      if postpos:
        postposies.append(postpos[-1])
  return postposies

sentences = get_chanks_by_sentence()

with open('./kotsugi/chapter05/knock45.txt', 'w+') as f:
  for chanks in sentences:
    for chank in chanks:
      text = ""
      if pred_morphs := chank.search_morph_by_pos("動詞"):
        # chankは動詞をふくむ文節

        pred_morph = pred_morph[0]
        postpos_morphs = search_postpos_morphs(chanks, chank.srcs)

        text += f"{pred_morph.base}\t"
        for postpos in postpos_morphs:
          text += f"{postpos.base} "
        
        text += "\n"
        print(text, end='')
        f.write(text)
        break