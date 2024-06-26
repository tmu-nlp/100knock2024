from knock41 import get_chanks_by_sentence

sentences = get_chanks_by_sentence()

for chanks in sentences:
  for chank in chanks:
    nx = chank.dst
    word = chank.marge_morphs()
    next_word = 'END'
    if nx != -1:
      next_word = chanks[nx].marge_morphs()

    print(f"{word}\t{next_word}")
