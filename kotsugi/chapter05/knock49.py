from knock41 import get_chanks_by_sentence, Chank

def make_path_list(chank: Chank, chanks: list[Chank], lst: list[int]):
  if chank.dst != -1:
    make_path_list(chanks[chank.dst], chanks, lst)
  lst.append(chank.srcs)
  return lst

def get_mearged_morphs(chanks: list[Chank], alias_target: list[int], count=0):
  lst = []
  for chank in chanks:
    if chank.srcs in alias_target:
      alias = 'X' if count == 1 else 'Y'
      lst.append(chank.aliased(alias))
      count += 1
    else:
      lst.append(chank.marge_morphs())
  return lst

def make_path(src: int, dst: int, chanks: list[Chank], alias_target: int, count = 1):
  path = get_mearged_morphs(chanks[src:dst+1], alias_target, count) 
  return ' -> '.join(path)

def print_pair_path(chank1: Chank, path1: list[int], chank2: Chank, path2: list[int], chanks: list[Chank]):
  if chank1.dst == -1:
    return
  if chank2.srcs in path1: # chank１からパスのリストにchank2のindex numberがあれば->で終了
    print(make_path(chank1.srcs, chank2.srcs, chanks, [chank1.srcs, chank2.srcs]))
  else:
    print(make_path(path1[0], path1[-2], chanks, [path1[0]]), end=' | ')
    print(make_path(path2[0], path2[-2], chanks, [path2[0]], 2), end=' | ')
    print(chanks[path1[-1]].marge_morphs())

sentences = get_chanks_by_sentence()

for chanks in sentences[56:57]:
  for chank1 in chanks:
    for chank2 in chanks[chank1.srcs:]:
      if (chank1.srcs == chank2.srcs):
        continue
      if chank1.include_pos('名詞') and chank2.include_pos('名詞'):
        path1 = list(reversed(make_path_list(chank1, chanks, [])))
        path2 = list(reversed(make_path_list(chank2, chanks, [])))
        print_pair_path(chank1, path1, chank2, path2, chanks)
        