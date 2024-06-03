import knock41
from graphviz import Digraph

num = 0 #何番目の文かを表しファイル名に入れる。

for sentence in knock41.sentences:
    dg = Digraph(format='png') #sentenceごとのグラフ
    for chunk in sentence.chunks:
        if chunk.dst != -1:
            modiin = []
            modifor = []
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    modiin.append(morph.surface)
            for morph in sentence.chunks[chunk.dst].morphs:
                if morph.pos != "記号":
                    modifor.append(morph.surface)
            phrasein = ''.join(modiin)
            phraseout = ''.join(modifor)
            dg.edge(phrasein, phraseout) #係り元のチャンクから係り先へのチャンクへとエッジを作る。
            # print(f"{phrasein}\t{phraseout}")
    dg.render('./44/' + str(num))
    num += 1