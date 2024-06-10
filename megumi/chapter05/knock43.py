"""
#43.名詞を含む文節が動詞を含む文節に係るものを抽出
名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．
ただし，句読点などの記号は出力しないようにせよ．
"""

for chunk in sentences[2].chunks:
  if int(chunk.dst) == -1:
    continue
  else:
    surf = "".join([morph.surface for morph in chunk.morphs if morph.pos != "記号"])
    next_surf = "".join([morph.surface for morph in sentences[2].chunks[int(chunk.dst)].morphs if morph.pos != "記号"]) 
    pos_noun = [morph.surface for morph in chunk.morphs if morph.pos == "名詞"]
    pos_verb = [morph.surface for morph in sentences[2].chunks[int(chunk.dst)].morphs if morph.pos == "動詞"]
    if pos_noun and pos_verb: 
      print(f"{surf}\t{next_surf}")
