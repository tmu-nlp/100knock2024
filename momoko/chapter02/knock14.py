basepath = "/Users/shirakawamomoko/Desktop/nlp100保存/chapter02/"
with open(basepath+"popular-names.txt", "r") as f:
    val = int(input("先頭から何行だけ表示しますか？："))
    lines = f.readlines()
    for i in range(val):
        lines[i] = lines[i].replace('\n', '')
        print(lines[i])