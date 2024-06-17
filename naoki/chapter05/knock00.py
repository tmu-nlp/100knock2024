import CaboCha
CBC = CaboCha.Parser()
divtext = []
with open("./ai.ja.txt", "r") as f, open("ai.ja.txt.parsed", "w") as f2:
  lines = f.readlines()
  for text in lines:
    if "。" in text:
      temp = text.split("。")
      temp = [x + "。" for x in temp if x != '']
      divtext.extend(temp)
  for text in divtext:
    tree = CBC.parse(text)
    f2.write(tree.toString(CaboCha.FORMAT_LATTICE))