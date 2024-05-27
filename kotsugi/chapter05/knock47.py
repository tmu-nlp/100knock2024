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
  prev_chank = chanks[0]
  for chank in chanks:
    text = ""
    if pred_morphs := chank.search_morph_by_pos("動詞"):
      pred_morph = pred_morphs[0]
      target_nouns = prev_chank.search_morph_by_pos("名詞", "サ変接続")
      target_nouns_txt = prev_chank.marge_morphs()

      if target_nouns and ('を' in target_nouns_txt):
        postpos_morphs, postpos_words = search_postpos_morphs(chanks, chank.srcs)

        text += f"{target_nouns_txt}{chank.marge_morphs()}\t"
        for postpos in postpos_morphs[:-1]:
          text += f"{postpos.base} "

        text += "\t"

        for postpos_word in postpos_words[:-1]:
          text += f"{postpos_word} "

        print(text)
        break
    prev_chank = chank