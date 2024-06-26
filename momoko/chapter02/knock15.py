import pandas as pd
basepath = "/Users/shirakawamomoko/Desktop/nlp100保存/chapter02/"
with open(basepath+"popular-names.txt", "r") as f:
    val = int(input("末尾から何行だけ表示しますか？："))
    lines = f.read().splitlines()#文字列を改行位置で分割している
    for i in reversed(range(val)):#reversed：逆順に参照できる．入力値->0．出力順を上からにするために．
        print(lines[len(lines)-1-i])#0の時に-1になるようにするための-1.