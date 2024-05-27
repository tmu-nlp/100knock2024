from knock41 import get_chanks_by_sentence, Chank

def make_path(chank: Chank, chanks: list[Chank], count):
  if chank.dst == -1 and count > 0:
    print("-> ", end='')
    print(f"{chank.marge_morphs()}")
    return
  elif chank.dst == -1:
    return
  elif count > 0:
    print("-> ", end='')
  
  print(f"{chank.marge_morphs()} ", end="")
  
  return make_path(chanks[chank.dst], chanks, count + 1)

sentences = get_chanks_by_sentence()

for chanks in sentences:
  for chank in chanks:
    if chank.include_pos('名詞'):
      make_path(chank, chanks, 0)
