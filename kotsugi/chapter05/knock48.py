from knock41 import get_chanks_by_sentence, Chank

def make_path(chank: Chank, chanks: list[Chank], count):
  if chank.dst == -1 and count > 0:
    # かかり先が終端だったら，表示して終了
    print("-> ", end='')
    print(chank.marge_morphs())
    return
  elif chank.dst == -1:
    # かかり先が終端かつ，分節の最後だったら，何も表示せず終了
    return
  elif count > 0:
    # かかり先が終端でなく，先頭でなかったら，矢印を表示
    print("-> ", end='')
  
  print(f"{chank.marge_morphs()} ", end="")
  
  return make_path(chanks[chank.dst], chanks, count + 1)

if __name__ == "__main__":
  sentences = get_chanks_by_sentence()

  for chanks in sentences:
    for chank in chanks:
      if chank.include_pos('名詞'):
        make_path(chank, chanks, 0)
