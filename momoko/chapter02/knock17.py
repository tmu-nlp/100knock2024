basepath = "/Users/shirakawamomoko/Desktop/nlp100保存/chapter02/"

with open(basepath+"col1.txt","r") as f:#col1を抽出したファイルを前のノックで作った!!
    lines = f.read().splitlines()
    new_lines = sorted(set(lines))#sorted:linesを昇順に並び替える．set:重複要素を消す．
    print(new_lines)
f.close()