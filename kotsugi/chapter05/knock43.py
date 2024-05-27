from knock41 import get_chanks_by_sentence

sentences = get_chanks_by_sentence()
  
for chanks in sentences:
  for chank in chanks:
    nx = chank.dst
    next_chank = chanks[nx]

    word = chank.marge_morphs()
    next_word = next_chank.marge_morphs()
    
    if nx == -1:
      continue

    if chank.include_pos('名詞') and next_chank.include_pos('動詞'):
      print(f"{word}\t{next_word}")
  