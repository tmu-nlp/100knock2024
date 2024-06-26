from knock40 import Morph
from knock41 import get_chanks_by_sentence, Chank

def search_postpos_morphs(chanks: list[Chank], dst: int) -> list[Morph]:
  postposies = []
  postposies_word = []
  for chank in chanks:
    if chank.dst == dst:
      postpos = chank.search_morph_by_pos("助詞")

      if postpos:
        postposies.append(postpos[-1])
        postposies_word.append(chank.marge_morphs())
  return postposies, postposies_word

sentences = get_chanks_by_sentence()

for chanks in sentences:
  for chank in chanks:
    text = ""
    if pred_morphs := chank.search_morph_by_pos("動詞"):
      pred_morph = pred_morphs[0]
      postpos_morphs, postpos_words = search_postpos_morphs(chanks, chank.srcs)

      text += f"{pred_morph.base}\t"
      for postpos in postpos_morphs:
        text += f"{postpos.base} "

      text += "\t"

      for postpos_word in postpos_words:
        text += f"{postpos_word} "

      print(text)
      break
        